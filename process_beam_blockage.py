#!/usr/bin/env python
# Compute beam blockage factor. Based off https://docs.wradlib.org/en/stable/notebooks/beamblockage/beamblockage.html
# see https://hess.copernicus.org/articles/17/863/2013/hess-17-863-2013.pdf (and cite it!)
# code is not working at the moment. Caching... 
import argparse
import multiprocessing
import sys
import typing
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

# for sites apart from Melbourne there are (small) changes in ht/location and beamwidth. Will need to deal with that.
# fortunately for me Melbourne is "easy"
STRM_dir = ausLib.data_dir / 'SRTM_Data'


def comp_strm(
        coords: Tuple[float, float],
        base_name: str,
        STRM_dir: pathlib.Path = STRM_dir,
        resoln: typing.Optional[list[int]] = None,
        cache: bool = True
        ) -> dict[str, xarray.DataArray]:

    if resoln is None:
        resoln = [90, 1000, 2000]
    files_to_make = {f'{r}m': STRM_dir / f'{base_name}_{int(r):d}m.tif' for r in resoln}
    result = dict()
    if cache:
        for res, file in files_to_make.items():
            if file.exists():
                result[res] = rioxarray.open_rasterio(file).squeeze().drop_vars('band')  # just load it up.
    if np.all([(k in result) for k in files_to_make.keys() ] ):
        # all the files exist. So save outselves some time and return the data.
        my_logger.info(f"DEM files {' '.join([str(f) for f in files_to_make.values()])} already exist\n Skipping further processing")
        return result
    files = list(STRM_dir.glob('*.hgt'))
    ds_raw = rioxarray.merge.merge_arrays([rioxarray.open_rasterio(f, parallel=True) for f in files])
    # and select within +/- 2 degrees of coords (roughly +/- 150 km) so we don't run out of memory.
    location = dict(x=coords[0], y=coords[1])
    ds_raw = ds_raw.sel(x=slice(location['x'] - 2, location['x'] + 2.),
                        y=slice(location['y'] + 2., location['y'] - 2.)
                        )
    ds_raw = ds_raw.where(ds_raw != ds_raw._FillValue, 0.0)  # Set missing values to zero.

    # make approx 90km, 1km & 2km resolution dataset
    # Note that STRM data is on long/lat grid while radar data is on "local" grid. Sigh!
    my_logger.debug(f"opened STRM dataset {ausLib.memory_use()}")
    for res, filename in files_to_make.items():
        if not ( cache and filename.exists() ):
            r=int(res[0:-1])
            coarse = np.round(r / 30).astype(int)
            ds = ds_raw.coarsen(x=coarse, y=coarse, boundary='pad').mean().load()
            ds.rio.to_raster(filename)
            my_logger.debug(f'{filename}, {coarse}, {ausLib.memory_use()}')
            result[f'{res}m'] = ds.squeeze().drop_vars('band')
            my_logger.debug(f'Created ds for {res}')


if __name__ == '__main__':
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!
    parser = argparse.ArgumentParser(description="Generate beam blockage and topog data for radar")
    parser.add_argument('site', type=str, help='site name for radar data')
    parser.add_argument('--output_dir', type=str, help='Base filename -- id_long_cbb_dem.nc will be added to this',
                        default=str(STRM_dir/'radar_strm'))
    parser.add_argument('--make_strm', action='store_true', help='Make STRM data if set. Note STRM data used')

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
    # possibly create the strm data. Quite slow...

    strm_info = comp_strm((metadata.site_lon.iloc[0], metadata.site_lat.iloc[0]),
                          base_name=site,cache=(not args.make_strm))
    for name, info in metadata.iterrows():
        outfile = pathlib.Path(args.output_dir) / f'{name}_cbb_dem.nc'
        # see if outfile exists and if we are allowed to overwrite it.
        if outfile.exists() and (not args.overwrite):
            my_logger.warning(f"Output file {outfile} exists and overwrite not set. Skipping further processing")
            continue
        outfile.parent.mkdir(exist_ok=True, parents=True)
        my_logger.info(f"Output file: {outfile}")
        sitecoords=[info.site_lon, info.site_lat, info.site_alt]
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

        ds = strm_info['90m'].sel(x=slice(rlimits[0], rlimits[2]), y=slice(rlimits[3], rlimits[1]))
        rastervalues = ds.values.squeeze()
        rastercoords = np.stack(np.meshgrid(ds.x, ds.y), axis=-1)
        crs = ds.rio.crs

        # Map rastervalues to polar grid points. Q ? Where does the cord ref system come in?
        polarvalues = wrl.ipol.cart_to_irregular_spline(
            rastercoords, rastervalues, polcoords, order=3, prefilter=False
            )
        proj_rad = wrl.georef.create_osr("aeqd",lat_0=info.site_lat, lon_0=info.site_lon, x_0=0., y_0=0.)
        proj_info=ausLib.gen_radar_projection(*sitecoords[0:2])
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
        # add in the proj info. A different name for eacj one.
        proj = xarray.DataArray(1).assign_attrs(proj_info)
        ds_grid[f'proj_{name}']=proj
        # add in the postchange_start as time.
        time_unit = 'minutes since 1970-01-01'  # units for time in output files
        ds_grid = ds_grid.assign_coords(postchange_start=info['postchange_start'])
        ds_grid.postchange_start.encoding.update(units=time_unit, dtype='float64')

        # add global attributes!
        site_meta_data = info.to_dict()
        site_meta_data.update(site=site,long_id=name)
        for k in site_meta_data.keys():
            if isinstance(site_meta_data[k], pd.Timestamp): # convert times to strings
                site_meta_data[k] = site_meta_data[k].strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(site_meta_data[k], pd._libs.tslibs.nattype.NaTType): # NaT
                site_meta_data[k]= '-'
            elif isinstance(site_meta_data[k], bool): # convert bools to ints.
                site_meta_data[k] = int(site_meta_data[k])

        ds_grid = ds_grid.assign_attrs(site_meta_data)
        ds_grid.to_netcdf(outfile,unlimited_dims='postchange_start')  # and save it
        my_logger.info(f"Saved {outfile}")
