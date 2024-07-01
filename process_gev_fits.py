#!/usr/bin/env python
# Compute the GEV fits -- uncertainties via sampling across events.
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
from ausLib import comp_radar_fit

# main code!
if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!

    parser = argparse.ArgumentParser(description="Compute GEV fits for radar event  data")
    parser.add_argument('input_file', type=pathlib.Path, help='input file of events from  radar data')
    parser.add_argument('--outdir', type=pathlib.Path,
                        help='output directory for fits. If not provided - computed from input file name'
                        )
    parser.add_argument('--nsamples', type=int, help='Number of samples to use', default=100)
    parser.add_argument('--bootstrap_samples', type=int, help='Number of event bootstraps to use', default=0)

    ausLib.add_std_arguments(parser)  # add on the std args
    args = parser.parse_args()
    my_logger = ausLib.process_std_arguments(args)  # setup the std stuff
    if args.outdir is None:
        output_dir = args.input_file.parent / 'fits'
    else:
        output_dir = args.output_dir
    my_logger.info(f"Output directory: {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)  # create dir f
    output_fit_t = output_dir / "gev_fit_temp.nc"
    output_fit = output_dir / "gev_fit.nc"
    output_summary = output_dir / "gev_summary.txt"
    output_fit_t_bs = output_dir / "gev_fit_temp_bs.nc"
    output_fit_bs = output_dir / "gev_fit_bs.nc"
    files = [output_fit, output_fit_t, output_summary]
    if args.bootstrap_samples > 0:
        my_logger.info(f"Bootstrapping")
        files.extend([output_fit_t_bs, output_fit_bs])
    exist = [file.exists() for file in files]
    if all(exist) and (not args.overwrite):
        my_logger.warning(f"All output files {files} exist and overwrite not set. Exiting")
        sys.exit(0)
    radar_dataset = xarray.load_dataset(args.input_file)  # load the processed radar
    threshold = 0.5  # some max are zero -- presumably no rain then.
    msk = (radar_dataset.max_value > threshold)
    radar_dataset = radar_dataset.where(msk)
    mn_temp = float(radar_dataset.ObsT.mean())  # FIXME -- when generating events store the mean temp.
    radar_dataset['Tanom'] = radar_dataset.ObsT - mn_temp  # make the anomaly and add it to the dataset.
    my_logger.info(f"Output files: fit_t: {output_fit_t}, fit: {output_fit} & summary: {output_summary}")
    extra_attrs = radar_dataset.attrs.copy()
    extra_attrs.update(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.utcnow().isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()]
                       )
    extra_attrs.update(mean_temp=mn_temp)

    my_logger.info(f"Computing fits")

    fit, fit_bs = comp_radar_fit(radar_dataset,
                                 n_samples=args.nsamples, bootstrap_samples=args.bootstrap_samples,
                                 name='fit_nocov', file=output_fit, bootstrap_file=output_fit_bs,
                                 extra_attrs=extra_attrs, recreate_fit=args.overwrite,
                                 )

    fit_t, fit_t_bs = comp_radar_fit(radar_dataset, cov=['Tanom'],
                                     n_samples=args.nsamples, bootstrap_samples=args.bootstrap_samples,
                                     extra_attrs=extra_attrs, name='fit_temp', file=output_fit_t,
                                     bootstrap_file=output_fit_t_bs,
                                     recreate_fit=args.overwrite,
                                     )
    my_logger.info(f"Computed fits")

    #
    def comp_dist(ds, samples=100):

        dist = scipy.stats.multivariate_normal(ds['mean'].squeeze(),
                                               ds['cov'].squeeze()
                                               )
        dist_sample = dist.rvs(size=samples)
        dist_sample = xarray.DataArray(data=dist_sample,
                                       coords=dict(sample=np.arange(0, args.nsamples), parameter=p_mean.parameter)
                                       )
        dist_sample = dist_sample.rename('cov_param_samp')
        return dist_sample
    with output_summary.open('w') as f:
        # print out the AIC
        q_aic = (fit_t.AIC - fit.AIC).quantile([0.1, 0.5, 0.9], dim='sample').to_dataframe().unstack()
        print(f"AIC: {q_aic.round(-1)}", file=f)
        if args.bootstrap_samples > 0:
            q_aic = (fit_t_bs.AIC - fit_bs.AIC).quantile([0.1, 0.5, 0.9], dim='bootstrap_sample').to_dataframe().unstack()
            print(f"AIC_bs: {q_aic.round(-1)}", file=f)
        # Uncertainties from the covariance matrix
        cov_mean = fit_t.Cov.mean('sample')
        p_mean = fit_t.Parameters.mean('sample')
        ds_cov = xarray.Dataset(dict(cov=cov_mean, mean=p_mean))
        samp_params = ds_cov.groupby('resample_prd').map(comp_dist,samples=args.nsamples)

        for p in ['location', 'scale']:
            dp = f'D{p}_Tanom'
            dfract = samp_params.sel(parameter=dp) / samp_params.sel(parameter=p)
            q = dfract.quantile([0.1, 0.5, 0.9], dim='sample').to_dataframe().unstack()
            print(f"   fract -cov  {dp}: {q.round(3)}", file=f)
            dfract_bs = fit_t_bs.Parameters.sel(parameter=dp) / fit_t_bs.Parameters.sel(parameter=p)
            q_bs = dfract_bs.quantile([0.1, 0.5, 0.9], dim='bootstrap_sample').to_dataframe().unstack()
            print(f"BS fract -cov  {dp}: {q_bs.round(3)}", file=f)
