#!/usr/bin/env python
# Compute the GEV fits -- just using the seasonal mean datasets and the acorn temperatures.
import multiprocessing
import pathlib
import sys
import typing

import numpy as np
import scipy.stats
import xarray
import ausLib

import argparse
import dask
import pandas as pd
from R_python import gev_r

# main code!
if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!

    parser = argparse.ArgumentParser(description="Compute GEV fits for radar data")
    parser.add_argument('input_file', type=pathlib.Path, help='input file of maxima from  radar data')
    parser.add_argument('--outdir', type=pathlib.Path,
                        help='output directory for fits. If not provided - computed from input file name'
                        )
    parser.add_argument('--nsamples', type=int, help='Number of samples to use', default=100)
    parser.add_argument('--station_id', type=int,
                        help='ACORN id for station used to generate temperature covariate. '
                             'If not provided computed from site in input file'
                        )
    parser.add_argument('--site', type=str, help='Site name -- overrides meta data in input files')

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

        #
    if args.outdir is None:
        output_dir = args.input_file.parent / 'fits'
    else:
        output_dir = args.output_dir
    my_logger.info(f"Output directory: {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)  # create dir f

    output_fit_t = output_dir / "gev_fit_raw_temp.nc"
    output_fit = output_dir / "gev_fit_raw.nc"
    output_summary = output_dir / "gev_summary_raw.txt"
    output_fit_t_bs = output_dir / "gev_fit_raw_temp_bs.nc"
    output_fit_bs = output_dir / "gev_fit_raw_bs.nc"
    files = [output_fit, output_fit_t, output_summary, output_fit_t_bs, output_fit_bs]
    file_exist = [file.exists() for file in files]
    if all(file_exist) and (not args.overwrite):
        my_logger.warning(f"All Output files {files} exist and overwrite not set. Exiting")
        sys.exit(0)
    radar_dataset = xarray.load_dataset(args.input_file)  # load the processed radar
    threshold = 0.5  # some max are zero -- presumably no rain then.
    msk = (radar_dataset.max_rain_rate > threshold)
    rain = radar_dataset.max_rain_rate.where(msk)

    extra_attrs = radar_dataset.attrs.copy()
    extra_attrs.update(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.utcnow().isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()]
                       )
    site = radar_dataset.attrs.get('site', None)
    if args.site:
        site = args.site
        extra_attrs.update(site=site)  # update the site
        my_logger.warning(f"Forcing site to be {site}")
    else:
        site = radar_dataset.attrs.get('site', None)
        my_logger.info(f"Site from metadata: {site}")
        if site is None:
            raise ValueError("No site name given or in metadata. Specify via --site option")

    # Get data for temperature covariate.
    station_id = args.station_id
    if station_id is None:
        station_id = ausLib.acorn_lookup[site]
        my_logger.debug('No station id provided. Using value from lookup table')
    my_logger.info(f'Using acorn id of {station_id}')
    # get the temperature data for the site.
    obs_temperature = ausLib.read_acorn(station_id, what='mean').resample('QS-DEC').mean()
    obs_temperature = obs_temperature.to_xarray().rename('ObsT').rename(dict(date='time'))
    obs_temperature = obs_temperature.where(obs_temperature.time.dt.season == 'DJF', drop=True)
    obs_temperature = obs_temperature.sel(time=radar_dataset.time, method='nearest')
    mn_temp = float(obs_temperature.mean())
    anomT = (obs_temperature - mn_temp).rename('Tanom')
    my_logger.info(f"Loaded temperature data for site {site}")
    # bootstrap-samples
    nsamples = args.nsamples
    rng = np.random.default_rng()
    locs = rng.integers(0, radar_dataset.time.size, size=(nsamples, radar_dataset.time.size))
    coords = dict(sample=np.arange(0, nsamples), index=np.arange(0, radar_dataset.time.size))
    locs = xarray.DataArray(locs,coords  = coords)


    rain_bs = rain.isel(time=locs)
    temp_bs = anomT.isel(time=locs)

    my_logger.info(f"Output files: fit_t: {output_fit_t}, fit: {output_fit} & summary: {output_summary}")
    my_logger.info(f"BS Output files: fit_t_bs: {output_fit_t_bs}, fit_bs: {output_fit_bs}")
    my_logger.info(f"Doing fits for {site} with {nsamples} samples")
    fit = gev_r.xarray_gev(rain, dim=['x', 'y', 'time'],
                           extra_attrs=extra_attrs, name='fit', file=output_fit, recreate_fit=args.overwrite
                           )

    fit_bs = gev_r.xarray_gev(rain_bs, dim=['x', 'y', 'index'],
                              extra_attrs=extra_attrs, name='fit_bs', file=output_fit_bs,
                              recreate_fit=args.overwrite,
                              )
    my_logger.info(f"Doing fits for {site} with {nsamples} samples and temp covariate")
    extra_attrs.update(mean_temp=mn_temp)
    fit_t = gev_r.xarray_gev(rain, dim=['x', 'y', 'time'], cov=anomT,
                             extra_attrs=extra_attrs, name='fit_temp', file=output_fit_t, recreate_fit=args.overwrite
                             )

    fit_t_bs = gev_r.xarray_gev(rain_bs, dim=['x', 'y', 'index'], cov=temp_bs,
                                extra_attrs=extra_attrs, name='fit_temp_bs', file=output_fit_t_bs,
                                recreate_fit=args.overwrite,
                                )
    # compute

    my_logger.info(f"Computed fits")

    # work out (using bs results) if the covariate changes are significant.
    # no check for overwriting as computations are cheap and if we have made fits then we want to update summary

    with (output_summary.open('w') as f):
        # 1) Print out the best estimate changes in the location and scale parameters.
        for p in ['location', 'scale']:
            dp = f'D{p}_Tanom'
            dfract = fit_t.Parameters.sel(parameter=dp) / fit_t.Parameters.sel(parameter=p)
            print(f"Best Est fract {dp}: {dfract.to_dataframe().round(3)}", file=f)
            # and the uncertainties in the change from the bootstrap estimates
            dfract_bs = fit_t_bs.Parameters.sel(parameter=dp) / fit_t_bs.Parameters.sel(parameter=p)
            dfract_bs = dfract_bs.quantile([0.1,0.5,0.9],dim='sample')
            print(f"BS estimates   {dp}: {dfract_bs.to_dataframe().unstack().round(3)}", file=f)
        # print out the AIC
        q_aic_bs = (fit_t_bs.AIC - fit_bs.AIC).quantile([0.1, 0.5, 0.9], dim='sample').to_dataframe().unstack()
        aic = (fit_t.AIC - fit.AIC).to_dataframe().unstack()
        print(f"Best Est AIC: {aic.round(0)}", file=f)
        print(f"BS est   AIC: {q_aic_bs.round(0)}", file=f)
