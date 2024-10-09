#!/usr/bin/env python
# extract gauge data within 150 km of radar site.
import pathlib
import argparse
import ausLib
import xarray
import numpy as np
import cartopy.crs as ccrs
import multiprocessing
import pandas as pd
if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!

    parser = argparse.ArgumentParser(description="Extract rainfall gauge data for radar sites")
    parser.add_argument('--input_dir', type=pathlib.Path, help='Where gauge data lives',
                        default = ausLib.agcd_rain_dir)
    parser.add_argument('--outdir', type=pathlib.Path,
                        help='output directory for extracted gauge data. ',
                        default=ausLib.data_dir / 'processed'
                        )
    parser.add_argument('--radius', type=float,
                        help='radius in km from radar site to extract gauge data',default=125.0)
    parser.add_argument('--calib', type=str, help='calibration to use', default='melbourne')

    ausLib.add_std_arguments(parser,dask=False)  # add on the std args. No dask as I/O task.
    args = parser.parse_args()
    my_logger = ausLib.process_std_arguments(args)  # setup the std stuff
    args.outdir.mkdir(parents=True, exist_ok=True)
    obs_files = sorted(args.input_dir.glob('*.nc'))
    ds_obs = xarray.open_mfdataset(obs_files).sel(time=slice('1997', None))

    r = args.radius * 1e3 # convert from km to m
    calib = args.calib
    rgn = dict(x=slice(-r, r), y=slice(-r, r))
    for site, no in ausLib.site_numbers.items():
        direct = ausLib.data_dir / f'processed/{site}_rain_{calib}'
        file = direct / f'monthly_mean_{site}_rain_{calib}.nc'
        if not file.exists():
            my_logger.warning(f'No file {file}')
            continue
        out_file = args.outdir / f'{site}_rain_{calib}'/f'gauge_rain_{site}_{calib}.nc'
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # check if outfile exists (and not got overwrite set)
        if (not args.overwrite) and out_file.exists():
            my_logger.info(f'File {out_file} exists. Skipping')
            continue

        ds = xarray.open_dataset(file)
        mean_rain = ds.mean_raw_rain_rate.sel(**rgn)

        # work  out region for gauge -- convert from m to long/lat
        radar_proj = ausLib.radar_projection(ds.proj.attrs)
        trans_coords = ccrs.PlateCarree().transform_points(radar_proj,
                                                           x=np.array([-r, r]), y=np.array([-r, r]))
        # get in the gauge data then inteprolate to radar grid and mask by radar data
        lon_slice = slice(trans_coords[0, 0], trans_coords[1, 0])
        lat_slice = slice(trans_coords[0, 1], trans_coords[1, 1])
        gauge = ds_obs.precip.sel(lon=lon_slice, lat=lat_slice)
        longitude = mean_rain.longitude.isel(time=0).squeeze(drop=True)
        latitude = mean_rain.latitude.isel(time=0).squeeze(drop=True)
        time = mean_rain.time
        gauge = gauge.interp(lon=longitude, lat=latitude, time=time).where(mean_rain.notnull()).load()
        # add some meta data to the gauge data

        extra_attrs = dict(program_name=str(pathlib.Path(__file__).name),
                           utc_time=pd.Timestamp.utcnow().isoformat(),
                           program_args=[f'{k}: {v}' for k, v in vars(args).items()],
                           site=site,
                           calib=calib,
                           radius=args.radius,
                           radar_file=str(file),
                           )

        gauge.attrs.update(extra_attrs)
        gauge.to_netcdf(out_file)
        my_logger.info(f'Wrote {out_file} for {site} using calib {calib}')
