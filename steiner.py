import typing
import warnings

import numpy as np
import xarray
from numba import jit, int32

"""
Code largely from Joshua Soderholm 
"""
warnings.simplefilter('ignore')

@jit(nopython=True)
def convective_radius(ze_bkg, area_relation):
    """
    Given a mean background reflectivity value, we determine via a step
    function what the corresponding convective radius would be.
    Higher background reflectivitives are expected to have larger
    convective influence on surrounding areas, so a larger convective
    radius would be prescribed.
    """
    if area_relation == 0:
        if ze_bkg < 30:
            conv_rad = 1000.
        elif (ze_bkg >= 30) & (ze_bkg < 35.):
            conv_rad = 2000.
        elif (ze_bkg >= 35.) & (ze_bkg < 40.):
            conv_rad = 3000.
        elif (ze_bkg >= 40.) & (ze_bkg < 45.):
            conv_rad = 4000.
        else:
            conv_rad = 5000.

    if area_relation == 1:
        if ze_bkg < 25:
            conv_rad = 1000.
        elif (ze_bkg >= 25) & (ze_bkg < 30.):
            conv_rad = 2000.
        elif (ze_bkg >= 30.) & (ze_bkg < 35.):
            conv_rad = 3000.
        elif (ze_bkg >= 35.) & (ze_bkg < 40.):
            conv_rad = 4000.
        else:
            conv_rad = 5000.

    if area_relation == 2:
        if ze_bkg < 20:
            conv_rad = 1000.
        elif (ze_bkg >= 20) & (ze_bkg < 25.):
            conv_rad = 2000.
        elif (ze_bkg >= 25.) & (ze_bkg < 30.):
            conv_rad = 3000.
        elif (ze_bkg >= 30.) & (ze_bkg < 35.):
            conv_rad = 4000.
        else:
            conv_rad = 5000.

    if area_relation == 3:
        if ze_bkg < 40:
            conv_rad = 0.
        elif (ze_bkg >= 40) & (ze_bkg < 45.):
            conv_rad = 1000.
        elif (ze_bkg >= 45.) & (ze_bkg < 50.):
            conv_rad = 2000.
        elif (ze_bkg >= 50.) & (ze_bkg < 55.):
            conv_rad = 6000.
        else:
            conv_rad = 8000.

    return conv_rad

@jit(nopython=True)
def peakedness(ze_bkg, peak_relation):
    """
    Given a background reflectivity value, we determine what the necessary
    peakedness (or difference) has to be between a grid point's
    reflectivity and the background reflectivity in order for that grid
    point to be labeled convective.
    """
    if peak_relation == 0:
        if ze_bkg < 0.:
            peak = 10.
        elif (ze_bkg >= 0.) and (ze_bkg < 42.43):
            peak = 10. - ze_bkg ** 2 / 180.
        else:
            peak = 0.

    elif peak_relation == 1:
        if ze_bkg < 0.:
            peak = 14.
        elif (ze_bkg >= 0.) and (ze_bkg < 42.43):
            peak = 14. - ze_bkg ** 2 / 180.
        else:
            peak = 4.

    return peak

@jit(nopython=True)
def steiner_classification(refl, x, y, dx, dy, intense=42, peak_relation=0,
                           area_relation=1, bkg_rad=11000, use_intense=True):
    """
    We perform the Steiner et al. (1995) algorithm for echo classification
    using only the reflectivity field in order to classify each grid point
    as either convective, stratiform or undefined. Grid points are classified
    as follows,

    0 = Undefined
    1 = Stratiform
    2 = Convective

    Parameters:
    ===========
    refl: ndarray
        Reflectivity slice (2D Cartesion grid).
    x: ndarray
        x-coordinates
    y: ndarray
        y-coordinates
    dx: float
        x-coordinates resolution
    dy: float
        y-coordinates resolution
    intense: float
        Value above which a pixel is consider convective no matter what.
    peak_relation : 0 or 1
        The peakedness relation.
    area_relation : 'small':0, 'medium':1, 'large':2, or 'sgp':3
        The convective area relation. See reference for more information.
    bkg_rad : float
        The background radius in meters. See reference for more information.
    use_intense : bool
        True to use the intensity criteria.
    Returns:
    ========
    sclass: ndarray <int>
        Convective/Stratiform classification, same size as refl.
    """
    
    sclass = np.zeros(refl.shape, dtype=int32)
    ny, nx = refl.shape

    for i in range(0, nx):
        # Get stencil of x grid points within the background radius
        imin = np.max(np.array([1, (i - bkg_rad / dx)], dtype=np.int32))
        imax = np.min(np.array([nx, (i + bkg_rad / dx)], dtype=np.int32))

        for j in range(0, ny):
            # First make sure that the current grid point has not already been
            # classified. This can happen when grid points within the
            # convective radius of a previous grid point have also been
            # classified.
            if ~np.isnan(refl[j, i]) & (sclass[j, i] == 0):
                # Get stencil of y grid points within the background radius
                jmin = np.max(np.array([1, (j - bkg_rad / dy)], dtype=np.int32))
                jmax = np.min(np.array([ny, (j + bkg_rad / dy)], dtype=np.int32))

                n = 0
                sum_ze = 0

                # Calculate the mean background reflectivity for the current
                # grid point, which will be used to determine the convective
                # radius and the required peakedness.

                for l in range(imin, imax):
                    for m in range(jmin, jmax):
                        if not np.isnan(refl[m, l]):
                            rad = np.sqrt(
                                (x[l] - x[i]) ** 2 + (y[m] - y[j]) ** 2)

                        # The mean background reflectivity will first be
                        # computed in linear units, i.e. mm^6/m^3, then
                        # converted to decibel units.
                            if rad <= bkg_rad:
                                n += 1
                                sum_ze += 10. ** (refl[m, l] / 10.)

                if n == 0:
                    ze_bkg = np.inf
                else:
                    ze_bkg = 10.0 * np.log10(sum_ze / n)

                # Now get the corresponding convective radius knowing the mean
                # background reflectivity.
                conv_rad = convective_radius(ze_bkg, area_relation)

                # Now we want to investigate the points surrounding the current
                # grid point that are within the convective radius, and whether
                # they too are convective, stratiform or undefined.

                # Get stencil of x and y grid points within the convective
                # radius.
                lmin = np.max(
                    np.array([1, np.int32(i - conv_rad / dx)], dtype=np.int32))
                lmax = np.min(
                    np.array([nx, np.int32(i + conv_rad / dx)], dtype=np.int32))
                mmin = np.max(
                    np.array([1, np.int32(j - conv_rad / dy)], dtype=np.int32))
                mmax = np.min(
                    np.array([ny, np.int32(j + conv_rad / dy)], dtype=np.int32))

                if use_intense and (refl[j, i] >= intense):
                    sclass[j, i] = 2

                    for l in range(lmin, lmax):
                        for m in range(mmin, mmax):
                            if not np.isnan(refl[m, l]):
                                rad = np.sqrt(
                                    (x[l] - x[i]) ** 2
                                    + (y[m] - y[j]) ** 2)

                                if rad <= conv_rad:
                                    sclass[m, l] = 2

                else:
                    peak = peakedness(ze_bkg, peak_relation)

                    if refl[j, i] - ze_bkg >= peak:
                        sclass[j, i] = 2

                        for l in range(imin, imax):
                            for m in range(jmin, jmax):
                                if not np.isnan(refl[m, l]):
                                    rad = np.sqrt(
                                        (x[l] - x[i]) ** 2
                                        + (y[m] - y[j]) ** 2)

                                    if rad <= conv_rad:
                                        sclass[m, l] = 2

                    else:
                        # If by now the current grid point has not been
                        # classified as convective by either the intensity
                        # criteria or the peakedness criteria, then it must be
                        # stratiform.
                        sclass[j, i] = 1

    return sclass


def main(refl_grid, x_vec, y_vec):


    dx = x_vec[1] - x_vec[0]
    dy = y_vec[1] - y_vec[0]
    sclass = steiner_classification(refl_grid, x_vec, y_vec, dx, dy)
    
    smeta = {'units': 'None',
            'long_name': 'Steiner echo classification',
            'valid_min': 0,
            'valid_max': 2,
            'description': ('Convective-stratiform echo '
                          'classification based on '
                          'Steiner et al. (1995) doi:https://doi.org/10.1175/1520-0450(1995)034<1978:CCOTDS>2.0.CO;2'),
            'comment':   ('0 = Undefined, 1 = Stratiform, '
                          '2 = Convective')}
    
    return sclass, smeta


def _steiner_one_slice(refl2d, x1d, y1d, dx, dy,area_relation:typing.Optional[typing.Literal[0,1,2,3]]=None):
    # Ensure raw ndarray/scalars for numba
    kwargs = dict(dx=float(dx), dy=float(dy))
    if area_relation is not None:
        kwargs['area_relation'] = area_relation
    refl2d = np.asarray(refl2d)
    x1d = np.asarray(x1d)
    y1d = np.asarray(y1d)
    out = steiner_classification(refl2d, x=x1d, y=y1d,**kwargs)
    return np.asarray(out, dtype=np.int16)  # adjust dtype if needed

def convert_to_m(coord):
    if coord.attrs['units'] == 'km':
        coord = coord*1000
        coord.attrs.update({'units': 'm'})
    return coord

def xarray_steiner(reflectivity:xarray.DataArray,
                   area_relation:typing.Optional[typing.Literal[0,1,2,3]] = None,
                   time_chunk:int = 2):
    """
    Do steiner classification
    :param reflectivity: reflectivity in Dbz
    Args:
        reflectivity: data array containing reflectivity in Dbz
        area_relation: See steiner_classification

    Returns:
    Steiner classified array. AI generated with guiding prompt
    """


    x = convert_to_m(reflectivity.x)
    y = convert_to_m(reflectivity.y)


    dx = float(x[1]-x[0])
    dy = float(y[1]-y[0])
    kwargs = {"dx": dx, "dy": dy}
    if area_relation != None:
        kwargs.update({"area_relation": area_relation})
    steiner_da = xarray.apply_ufunc(
        _steiner_one_slice,
        reflectivity.chunk({"valid_time": time_chunk, "y": -1, "x": -1}),
        x, y,
        input_core_dims=[["y", "x"], ["x"], ["y"]],
        output_core_dims=[["y", "x"]],
        kwargs=kwargs,
        vectorize=True,          # loops over valid_time
        dask="parallelized",
        output_dtypes=[np.int32], # lowest precision we can do.
        keep_attrs=True,
    ).rename("steiner_class")


    smeta = {'units': 'None',
            'long_name': 'Steiner echo classification',
            'valid_min': 0,
            'valid_max': 2,
             "dx": dx,
             "dy": dy,
            'description': ('Convective-stratiform echo '
                          'classification based on '
                          'Steiner et al. (1995) doi:https://doi.org/10.1175/1520-0450(1995)034<1978:CCOTDS>2.0.CO;2'),
            'comment':   ('0 = Undefined, 1 = Stratiform, '
                          '2 = Convective')}

    steiner_da.attrs.update(smeta)

    return steiner_da.persist()

_NUMBA_WARMED = False

def warmup_steiner(dtype=np.float64):
    global _NUMBA_WARMED
    if _NUMBA_WARMED:
        return

    dummy_refl = np.zeros((16, 16), dtype=dtype)
    dummy_x = np.arange(16, dtype=dtype) * 500.0
    dummy_y = np.arange(16, dtype=dtype) * -500.0
    _ = steiner_classification(dummy_refl, dummy_x, dummy_y, 500.0, -500.0)

    _NUMBA_WARMED = True

if __name__ == "__main__":
    import ausLib#
    import pathlib
    import multiprocessing
    import datetime
    # guesses for optimum behaviour
    ncores =8
    time_chunk = 2
    warmup = True
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!
    client = ausLib.dask_client(mode='cpu',max_cores=ncores) # cpu mode rather than I/O mode.
    if warmup:
        warmup_steiner()

    refl = ausLib.read_radar_zipfile(pathlib.Path(
        r"C:\Users\stett2\OneDrive - University of Edinburgh\data\aus_radar_analysis\radar\raw_radar_data\hist_gndrefl\46\2015\46_20150125.gndrefl.zip"))
    print("Starting steiner classification")
    start_time = datetime.datetime.now()
    steiner_class = xarray_steiner(refl.reflectivity,time_chunk=time_chunk,area_relation=0)
    steiner_class.load()
    end_time = datetime.datetime.now()
    delta_time = end_time - start_time
    dsec = np.round(delta_time.total_seconds(),1)
    print(f"Took {dsec} secs to run Steiner classification on {len(client.scheduler_info()['workers'])} cores with warmup={warmup}")
    # shut down the dask local cluster
    client.shutdown()
    client.close()