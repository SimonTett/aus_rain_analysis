#!/usr/bin/env python
# Compute the GEV fits
import multiprocessing
import pathlib
import sys
import typing

import numpy as np
import scipy.stats
import xarray
import ausLib
from R_python import gev_r
from numpy import random
import argparse
import dask
import pandas as pd


def comp_radar_fit(
        dataset: xarray.Dataset,
        cov: typing.Optional[list[xarray.DataArray]] = None,
        n_samples: int = 100,
        rng_seed: int = 123456,
        file: typing.Optional[pathlib.Path] = None,
        name: typing.Optional[str] = None,
        extra_attrs: typing.Optional[dict] = None
) -> xarray.Dataset:
    """
    Compute the GEV fits for radar data. This is a wrapper around the R code to do the fits.
    Args:
        dataset: dataset
        cov: List of covariate data arrays
        n_samples: number of samples for random selection
        rng_seed: seed for random number generator
        file: file for output
        name:  name of output file
        extra_attrs: any extra attributes to be added

    Returns:fit

    """
    rng = random.default_rng(rng_seed)
    rand_index = rng.integers(len(dataset.quantv), size=n_samples)
    coord = dict(sample=np.arange(0, n_samples))
    ds = dataset.isel(quantv=rand_index). \
        rename(dict(quantv='sample')).assign_coords(**coord)
    cov_rand = None
    if cov is not None:
        cov_rand = [c.isel(quantv=rand_index).rename(dict(quantv='sample')).assign_coords(**coord) for c in cov]

    wt = ds.count_cells
    mx = ds.max_value

    fit = gev_r.xarray_gev(mx, cov=cov_rand, dim='EventTime', weights=wt,
                           recreate_fit=True, file=file, name=name, extra_attrs=extra_attrs
                           )
    return fit


# main code!
if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!

    parser = argparse.ArgumentParser(description="Compute GEV fits for radar data")
    parser.add_argument('input_file', type=str, help='input file of events from  radar data')
    parser.add_argument('output_root', type=str,
                        help='Root file name where fits and summary info  goes'
                        )
    parser.add_argument('--nsamples', type=int, help='Number of samples to use', default=100)
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
    output_fit_t = pathlib.Path(f"{args.output_root}_fit_temp.nc")
    output_fit = pathlib.Path(f"{args.output_root}_fit.nc")
    output_summary = pathlib.Path(f"{args.output_root}_summary.txt")
    for file in [output_fit, output_fit_t, output_summary]:
        if file.exists() and (not args.overwrite):
            my_logger.warning(f"Output file {file} exists and overwrite not set. Exiting")
            sys.exit(0)
    radar_dataset = xarray.load_dataset(args.input_file)  # load the processed radar
    threshold = 0.5  # some max are zero -- presumably no rain then.
    msk = (radar_dataset.max_value > threshold)
    radar_dataset = radar_dataset.where(msk)
    mn_temp = float(radar_dataset.ObsT.mean())

    output_summary.parent.mkdir(exist_ok=True, parents=True)
    my_logger.info(f"Output files: fit_t: {output_fit_t}, fit: {output_fit} & summary: {output_summary}")
    extra_attrs = dict(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.utcnow().isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()]
                       )
    fit = comp_radar_fit(radar_dataset, n_samples=args.nsamples,
                         name='fit_nocov', file=output_fit, extra_attrs=extra_attrs
                         )
    extra_attrs.update(mean_temp=mn_temp)
    fit_t = comp_radar_fit(radar_dataset, cov=[(radar_dataset.ObsT - mn_temp).rename('Tanom')],
                           n_samples=args.nsamples,
                           extra_attrs=extra_attrs, name='fit_temp', file=output_fit_t
                           )

    ## work out (in two different ways if the covariate changes are significant.

    with output_summary.open('w') as f:
        # 1) Compute 10 & 90% quantiles for location and scale parameters from random samples
        for p in ['location', 'scale']:
            dp = f'D{p}_Tanom'
            dfract = fit_t.Parameters.sel(parameter=dp) / fit_t.Parameters.sel(parameter=p)
            q = dfract.quantile([0.1, 0.5, 0.9], dim='sample').to_dataframe().unstack()
            print(f"fract {dp}: {q.round(3)}", file=f)
        # print out the AIC
        q_aic = (fit_t.AIC - fit.AIC).quantile([0.1, 0.5, 0.9], dim='sample').to_dataframe().unstack()
        print(f"AIC: {q_aic.round(3)}", file=f)
        #2 use the mean covariance matrix.
        cov_mean = fit_t.Cov.mean('sample')
        p_mean = fit_t.Parameters.mean('sample')
        samp_params = []
        for resample in p_mean.resample_prd:
            dist = scipy.stats.multivariate_normal(p_mean.sel(resample_prd=resample),
                                                   cov_mean.sel(resample_prd=resample)
                                                   )
            dist_sample = dist.rvs(size=args.nsamples)
            dist_sample = xarray.DataArray(data=dist_sample,
                                           coords=dict(sample=np.arange(0, 100), parameter=p_mean.parameter)
                                           )
            dist_sample = dist_sample.assign_coords(resample_prd=resample).rename('cov_param_samp')
            samp_params.append(dist_sample)
        samp_params = xarray.concat(samp_params, dim='resample_prd')
        for p in ['location', 'scale']:
            dp = f'D{p}_Tanom'
            dfract = samp_params.sel(parameter=dp) / samp_params.sel(parameter=p)
            q = dfract.quantile([0.1, 0.5, 0.9], dim='sample').to_dataframe().unstack()
            print(f"fract -cov  {dp}: {q.round(3)}", file=f)
