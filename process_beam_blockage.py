#!/usr/bin/env python
# Compute beam blockage factor. Based off https://docs.wradlib.org/en/stable/notebooks/beamblockage/beamblockage.html
# see https://hess.copernicus.org/articles/17/863/2013/hess-17-863-2013.pdf (and cite it!)
# Note that code uses a lot of memory... Will generate regional DEM files and then CBB_DEM for each sub-site
import argparse
import multiprocessing
import sys
import typing
import zipfile
from typing import Tuple

import dask
import pandas as pd
import wradlib as wrl

import xarray
import xradar
import warnings
import osgeo
import numpy as np
import pathlib
import rioxarray
import rioxarray.merge
import cartopy.crs as ccrs
import ausLib
import earthaccess

SRTM_dir = ausLib.data_dir / 'SRTM_Data'


def get_dem_files(bounding_box: tuple[float, float, float, float],
                  cache_dir: typing.Optional[pathlib.Path] = SRTM_dir,
                  dryrun: bool = False) -> list[pathlib.Path]:
    """
    Get the DEM files that cover the bounding box.

    :param bounding_box: Co-ordinate bounds for DEM files. Format is (min_lon, min_lat, max_lon, max_lat)
    :param dryrun: do a dryrun. No files will be downloaded though earrhaccess will be queried.
    :param cache_dir: Where downloaded STRM files will be stored.
    :return: a list of files that should cover the bounding box.
    Data will be retrieved if zip files do not exist in the STRM directory.
    """
    if not (isinstance(bounding_box, tuple) and len(bounding_box) == 4):
        raise ValueError('Bounding box should be a tuple of 4 floats')
    cache_dir.mkdir(exist_ok=True, parents=True) # create cache dir if needed.
    # use earthaccess to get the files.
    stat = earthaccess.login(strategy='netrc')  # if userid etc not found this still works. So check
    if not stat.authenticated:
        raise ValueError('Could not authenticate with earthdata. Check your netrc file. Create with '
                         'create_earthdata_files.py')
    lst_data = earthaccess.search_data(short_name='SRTMGL1', bounding_box=bounding_box)
    # what data to we already have?
    data_to_retrieve = []
    files = []  # list of files that are in the bounding box. We will return this.
    for data in lst_data:
        # work out what files are in the granule. After download and unziping
        files_in_granule = [cache_dir / pathlib.Path(lnk).name.replace('.SRTMGL1.hgt.zip', '.hgt')
                            for lnk in data.data_links()]
        files += files_in_granule
        if not all([f.exists() for f in files_in_granule]):
            data_to_retrieve.append(data)
            my_logger.debug(f"Need to retrieve {data}")

    n_granules = len(data_to_retrieve)
    if n_granules > 0:
        if dryrun:
            print(f'Would download {n_granules} granules')
            for g in data_to_retrieve:
                my_logger.info(g)
        else:
            my_logger.info(f"Retrieving {len(data_to_retrieve)} granules of SRTM data.")
            files_retrieved = earthaccess.download(data_to_retrieve, cache_dir)
            # convert to pathlib and check exists
            files_retrieved = [pathlib.Path(f) for f in files_retrieved]

            # now to unzip the data.
            for file in files_retrieved:
                if not file.exists():
                    ValueError(f"File {file} does not exist. Failed to download.")
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(cache_dir)
                    my_logger.debug(f'Extracted {file}')
    else:
        my_logger.info('All needed files already exist')
    return files


def comp_strm(
        bounding_box: Tuple[float, float, float, float],
        base_name: str,
        srtm_dir: pathlib.Path = SRTM_dir,
        resoln: typing.Optional[list[int]] = None,
        cache: bool = True
) -> dict[str, xarray.DataArray]:
    if resoln is None:
        resoln = [90, 1000, 2000]
    files_to_make = {f'{r}m': srtm_dir / f'{base_name}_{int(r):d}m.tif' for r in resoln}
    result = dict()
    if cache:
        for res, file in files_to_make.items():
            if file.exists():
                result[res] = rioxarray.open_rasterio(file).squeeze().drop_vars('band')  # just load it up.
    if np.all([(k in result) for k in files_to_make.keys()]):
        # all the files exist. So save ourselves some time and return the data.
        my_logger.info(
            f"DEM files {' '.join([str(f) for f in files_to_make.values()])} already exist\n Skipping further processing")
        return result
    files = get_dem_files(bounding_box=bounding_box, cache_dir=srtm_dir)
    my_logger.debug('Opening up SRTM data')
    ds_raw = rioxarray.merge.merge_arrays([rioxarray.open_rasterio(f, parallel=True) for f in files])

    ds_raw = ds_raw.sel(x=slice(bounding_box[0], bounding_box[2]),
                        y=slice(bounding_box[3], bounding_box[1]) # y coords are reversed.
                        )
    ds_raw = ds_raw.where(ds_raw != ds_raw._FillValue, 0.0)  # Set missing values to zero.

    # make approx 90km, 1km & 2km resolution dataset
    # Note that SRTM data is on long/lat grid while radar data is on "local" grid. Sigh!
    my_logger.debug(f"opened SRTM dataset {ausLib.memory_use()}")
    for res, filename in files_to_make.items():
        if not (cache and filename.exists()):
            r = int(res[0:-1])
            coarse = np.round(r / 30).astype(int)
            ds = ds_raw.coarsen(x=coarse, y=coarse, boundary='pad').mean().load()
            ds.rio.to_raster(filename)
            my_logger.debug(f'{filename}, {coarse}, {ausLib.memory_use()}')
            result[f'{res}'] = ds.squeeze().drop_vars('band')
            my_logger.debug(f'Created ds for {res}')
    return result


if __name__ == '__main__':
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!
    parser = argparse.ArgumentParser(description="Generate beam blockage and topog data for radar")
    parser.add_argument('site', type=str, help='site name for radar data')
    parser.add_argument('--output_dir', type=pathlib.Path, help='Output directory for CB/DEM data')
    parser.add_argument('--make_srtm', action='store_true',
                        help='Remake SRTM data if set otherwise cache data. Note STRM data used to generate DEM data')
    parser.add_argument('--srtm_cache_dir',type=pathlib.Path, default=SRTM_dir,
                        help='Where SRTM data downloaded from earthdata is stored')

    ausLib.add_std_arguments(parser)  # add on the std args
    args = parser.parse_args()
    my_logger = ausLib.setup_log(args.verbose, log_file=args.log_file)  # setup the logging
    for name, value in vars(args).items():
        my_logger.info(f"Arg:{name} =  {value}")
    if args.dask:
        my_logger.info('Starting dask client')
        client = ausLib.dask_client()
    else:
        dask.config.set(scheduler="single-threaded")  # make sure dask is single threaded.
        my_logger.info('Running single threaded')

    # open up the metadata file

    #  get the co-ords.
    site = args.site
    metadata = ausLib.site_info(ausLib.site_numbers[site])

    for name, info in metadata.iterrows():
        outfile = pathlib.Path(args.output_dir) / f'{name}_cbb_dem.nc'
        outfile.parent.mkdir(exist_ok=True, parents=True)
        # see if outfile exists and if we are allowed to overwrite it.
        if outfile.exists() and (not args.overwrite):
            my_logger.warning(f"Output file {outfile} exists and overwrite not set. Skipping further processing")
            continue
        my_logger.info(f"Output file: {outfile}")
        sitecoords =   (info.site_lon, info.site_lat, info.site_alt)
        nrays = int(np.ceil(360 / info.beamwidth))  # number of rays
        nbins = 400  # number of range bins
        el = info.beamwidth / 2.  # vertical antenna pointing angle (deg)
        bw = info.beamwidth  # half power beam width (deg) -- check with Joshua
        range_res = 500.0  # range resolution (meters)
        r = np.arange(nbins) * range_res
        beamradius = wrl.util.half_power_radius(r, bw)
        coord = wrl.georef.sweep_centroids(nrays, range_res, nbins, el)
        coords = wrl.georef.spherical_to_proj(
            coord[..., 0], coord[..., 1], coord[..., 2],
            sitecoords
        )
        lon = coords[..., 0]
        lat = coords[..., 1]
        alt = coords[..., 2]
        polcoords = coords[..., :2]
        my_logger.debug(f"lon,lat,alt: {coords.shape}")
        rlimits = (lon.min(), lat.min(), lon.max(), lat.max())
        my_logger.info(
            f"Radar bounding box:\n\t{rlimits[3]:.2f}\n{rlimits[0]:.2f}      {rlimits[2]:.2f}\n\t{rlimits[1]:.2f}"
        )
        # possibly create the strm data. Quite slow...
        delta=0.5
        relimits_broad=(rlimits[0]-delta,rlimits[1]-delta,rlimits[2]+delta,rlimits[3]+delta)
        ds = comp_strm(bounding_box=rlimits, base_name=site, cache=(not args.make_srtm))['90m']
        rastervalues = ds.values.squeeze()
        rastercoords = np.stack(np.meshgrid(ds.x, ds.y), axis=-1)
        crs = ds.rio.crs

        # Map rastervalues to polar grid points. Q ? Where does the cord ref system come in?
        polarvalues = wrl.ipol.cart_to_irregular_spline(
            rastercoords, rastervalues, polcoords, order=3, prefilter=False
        )
        proj_rad = wrl.georef.create_osr("aeqd", **dict(lat_0=info.site_lat, lon_0=info.site_lon, x_0=0., y_0=0.))
        proj_info = ausLib.gen_radar_projection(*sitecoords[0:2])
        DEM = wrl.georef.create_xarray_dataarray(
            polarvalues, r=r, phi=coord[:, 0, 1], site=sitecoords).wrl.georef.georeference(crs=proj_rad)

        PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
        PBB = np.ma.masked_invalid(PBB)
        CBB = wrl.qual.cum_beam_block_frac(PBB)
        CBB = wrl.georef.create_xarray_dataarray(
            CBB, r=r, phi=coord[:, 0, 1], site=sitecoords, theta=el,
        ).wrl.georef.georeference(crs=proj_rad)

        # and regrid -- coords from inspection of the Melbourne ground reflectivity.
        cart = xarray.Dataset(coords={"x": (["x"], np.arange(-127.75e3, 127.5e3, 500)),
                                      "y": (["y"], np.arange(-127.75e3, 127.5e3, 500))}
                              )
        src = np.stack([CBB.x.values.flatten(), CBB.y.values.flatten()], axis=-1)
        trg = np.meshgrid(cart.x.values, cart.y.values)
        trg = np.vstack((trg[0].ravel(), trg[1].ravel())).T
        interpol = wrl.ipol.OrdinaryKriging  # default krieging.
        #interpol = wrl.ipol.OrdinaryKriging(src,trg)

        CBB_grid = CBB.wrl.comp.togrid(cart, radius=200e3, center=(0, 0), interpol=interpol)
        DEM_grid = DEM.wrl.comp.togrid(cart, radius=200e3, center=(0, 0), interpol=interpol)
        ds_grid = xarray.Dataset(dict(elevation=DEM_grid, CBB=CBB_grid))
        # add in the proj info. A different name for each one.
        proj = xarray.DataArray(1).assign_attrs(proj_info)
        ds_grid[f'proj_{name}'] = proj
        # add in the postchange_start as time.
        time_unit = 'minutes since 1970-01-01'  # units for time in output files
        ds_grid = ds_grid.assign_coords(postchange_start=info['postchange_start'])
        ds_grid.postchange_start.encoding.update(units=time_unit, dtype='float64')

        # add global attributes!
        site_meta_data = info.to_dict()
        site_meta_data.update(site=site, long_id=name)
        for k in site_meta_data.keys():
            if isinstance(site_meta_data[k], pd.Timestamp):  # convert times to strings
                site_meta_data[k] = site_meta_data[k].strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(site_meta_data[k], pd._libs.tslibs.nattype.NaTType):  # NaT
                site_meta_data[k] = '-'
            elif isinstance(site_meta_data[k], bool):  # convert bools to ints.
                site_meta_data[k] = int(site_meta_data[k])

        ds_grid = ds_grid.assign_attrs(site_meta_data)
        ds_grid.to_netcdf(outfile, unlimited_dims='postchange_start')  # and save it
        my_logger.info(f"Saved {outfile}")
