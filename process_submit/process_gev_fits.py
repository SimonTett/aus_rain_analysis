#!/usr/bin/env python
# Compute the GEV fits -- uncertainties via sampling across events.
import logging
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

from numpy import random
my_logger=logging.Logger('gev_fits')

def sample_events2(
        data_set: xarray.Dataset,
        rng: np.random.BitGenerator = None,
        dim: str = 'EventTime',

) -> xarray.Dataset:
    """
    Sample events with replacement from a data array. Not very generic...
    Args:
        data_set: xarray data set
        nsamples: number of bootstrap samples to take
        rng: random number generator
        dim: dimension over which to apply dropna and sampling.
        dim2: dim over which to randomly sample.
    Returns: xarray data array
    """
    if rng is None:
        rng = np.random.default_rng()
        my_logger.debug('RNG initialised')
    my_logger.log(1, 'Starting sampling')
    # hard wired but hopefully fast.
    sel_dim = {d: 0 for d in data_set.dims if d != dim}
    indx = data_set.max_value.isel(**sel_dim).notnull().compute().squeeze(drop=True)  # Mask is constant.
    indx = indx.drop_vars(sel_dim.keys(), errors='ignore')
    ds = data_set.where(indx, drop=True)  # drop the NaNs
    # npts = indx.sum()
    # ds = data_set.dropna(dim)
    my_logger.log(1, 'Dropped NaNs')
    npts = ds.EventTime.size
    locs = rng.integers(0, npts, size=(npts))  # single bootstrap samples.
    # wrap it up in  data array to allow fancy indexing
    coords = dict(index=np.arange(0, npts))
    locs = xarray.DataArray(locs, coords=coords)
    my_logger.log(1, 'Computed indices')
    ds = ds.isel({dim: locs})  # and sample at dim
    my_logger.log(1, 'Sampled data')
    return ds


def comp_radar_fit(
        dataset: xarray.Dataset,
        cov: typing.Optional[typing.Union[list[str] , str]] = None,
        n_samples: int = 100,
        bootstrap_samples: int = 0,
        rng_seed: int = 123456,
        file: typing.Optional[pathlib.Path] = None,
        bootstrap_file: typing.Optional[pathlib.Path] = None,
        recreate_fit: bool = False,
        name: typing.Optional[str] = None,
        extra_attrs: typing.Optional[dict] = None,
        use_dask: bool = False,
        verbose: bool = False
) -> tuple[xarray.Dataset, typing.Optional[xarray.Dataset]]:
    """
    Compute the GEV fits for radar data. This is a wrapper around the R code to do the fits.
    It can do various dask things. However, when using dask there is a risk of deadlock.
    If so. rerun wth more memory.
    Args:

        dataset: dataset
        cov: List of covariate names. Extracted from dataset.
        n_samples: number of samples for random selection
        bootstrap_samples:  Number of samples for bootstrap calculation. If 0 no bootstrap is done.
        rng_seed: seed for random number generator
        file: file for output
        bootstrap_file: file for bootstrap output
        recreate_fit: if True recreate the fit
        name:  name of dataset. _bs will be appended for bootstrap fit
        extra_attrs: any extra attributes to be added
        use_dask: If True use dask for the computation.
        verbose: Passed through to R code. Will be veryyy verbose.

    Returns:fit

    """
    # doing this at run time as R is not always available.

    from R_python import gev_r
    if isinstance(cov, str):
        cov = [cov]  # convert it to a list.
    if (file is not None) and file.exists() and (
            not recreate_fit):  # got a file specified, it exists and we are not recreating fit
        fit = xarray.load_dataset(file)  # just load the dataset
        if (bootstrap_file is not None) and bootstrap_file.exists() and (
                not recreate_fit):  # got a file specified, it exists and we are not recreating fit
            bs_fit = xarray.load_dataset(bootstrap_file).mean('sample')  # just load the dataset
            return fit, bs_fit  # can now return the fit and bootstrap fit

    # normal stuff
    rng = random.default_rng(rng_seed)
    rand_index = rng.integers(1, len(dataset.quantv) - 2, size=n_samples)  # don't want the min or max quantiles
    coord = dict(sample=np.arange(0, n_samples))
    # randomly select from quantiles for each event -- they all exist (or not)
    ds = dataset.isel(quantv=rand_index).\
        rename(dict(quantv='sample')).assign_coords(**coord)
    if use_dask:
        my_logger.debug(f'Chunking data for best est  {ausLib.memory_use()}')
        samp_chunk = max(n_samples // 20, 1)
        samp_chunk = 4
        ds = ds.chunk(sample=samp_chunk, resample_prd=-1, EventTime=-1)  # parallelize over sample
        my_logger.debug(f'Chunk size is: {ds.chunksizes}')
        my_logger.debug(f'Rechunked data for best est  {ausLib.memory_use()}')
    cov_rand = None
    if cov is not None:
        cov_rand = [ds[c] for c in cov]

    wt = ds.count_cells
    mx = ds.max_value
    my_logger.debug('Doing GEV fit')
    fit = gev_r.xarray_gev(mx, cov=cov_rand, dim='EventTime', weights=wt, verbose=verbose,
                           recreate_fit=recreate_fit, file=file, name=name, extra_attrs=extra_attrs,
                           use_dask=use_dask)
    my_logger.debug('Done GEV fit')

    if bootstrap_samples > 0:
        if bootstrap_file and bootstrap_file.exists() and (not recreate_fit):
            bs_fit = xarray.load_dataset(bootstrap_file)  # load the file
            my_logger.debug(f"Loaded Bootstrap fits {ausLib.memory_use()}")
        else:
            my_logger.info(f"Calculating bootstrap for {name} {ausLib.memory_use()}")
            # loop over bootstrap samples and then concat together at the end.
            # generating all the samples produces a very large mem object and requires data
            # to be moved around. So don't bother...
            bs_sample_fits = []  # where the individual samples go.
            init_params = fit.median('sample').Parameters
            # get the initial parameters. Speeds up the fit by about 10-20% and hopefully reduces odds of bad fits...
            for samp in range(bootstrap_samples):
                my_logger.debug('Bootstrapping')
                ds_bs = ds.groupby('resample_prd',squeeze=False).map(sample_events2, rng=rng,
                                                       dim='EventTime'). \
                    drop_vars('EventTime').assign_coords(bootstrap_sample=samp)
                my_logger.debug('Done bootstrapping. Extracting covars')
                cov_rand = None
                if cov is not None:
                    cov_rand = [ds_bs[c] for c in cov]
                my_logger.debug(f'Doing BS GEV fit for sample {samp}')
                bs_fit = gev_r.xarray_gev(ds_bs.max_value, cov=cov_rand, dim='index', weights=ds_bs.count_cells,
                                          verbose=verbose, use_dask=use_dask,initial=init_params)
                bs_sample_fits.append(bs_fit)
                my_logger.info(f'Done BS fit for sample {samp} {ausLib.memory_use()}')
            # all done doing the individual fits. Now need to concat them together.
            bs_fit = xarray.concat(bs_sample_fits, dim='bootstrap_sample')

            if extra_attrs:
                bs_fit.attrs.update(extra_attrs)
            if bootstrap_file:
                bs_fit.to_netcdf(bootstrap_file)
            bs_fit = bs_fit.mean('sample')



    else:
        bs_fit = None

    return fit, bs_fit
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
    parser.add_argument('--time_range', type=pd.Timestamp, nargs='+',
                        help='Time range to use for fits. Provide 1 or 2 arguments', default=None)
    ausLib.add_std_arguments(parser)  # add on the std args
    args = parser.parse_args()
    if args.outdir is None:
        parent = args.input_file.parent
        output_dir = parent / 'fits'
    else:
        output_dir = args.outdir
        #base_dir = output_dir
    ## defaults for submission.
    pbs_log_dir = output_dir/ 'pbs_logs'
    run_log_dir = output_dir / 'run_logs'
    log_file = 'gev_fits'+pd.Timestamp.utcnow().strftime('%Y_%m_%d_%H%M%S')
    log_base_default = pbs_log_dir /log_file
    log_file_default =  run_log_dir/f'{log_file}.log'
    job_name_default=f'gev_{output_dir.stem[0:10]}'
    json_file_default = ausLib.module_path/'config_files/process_gev_fits.json'
    default = dict(log_base=log_base_default, #where pbs logs (stdout & stderr)
                   log_file=log_file_default, # log while run is going
                   job_name=job_name_default, # default name of job
                   json_submit_file=json_file_default)
    my_logger = ausLib.process_std_arguments(args,default=default)  # setup the std stuff

    use_dask = args.dask
    if use_dask:
        raise NotImplementedError("dask  and rpy2 do not work well together")
    # check times are 1 or 2 element. If 1 element then assume it is the start time.
    time_range = None
    if args.time_range is not None:
        if len(args.time_range) == 1:
            time_range = (args.time_range[0],None)
            out_post = args.time_range[0].strftime('_%Y%m%d')
        elif len(args.time_range) == 2:
            time_range = tuple(args.time_range)
            out_post = args.time_range[0].strftime('_%Y%m%d')
        else:
            parser.print_help()
            raise ValueError("Time range must be 1 or 2 elements")

    my_logger.debug(f'use_dask is {use_dask}')
    my_logger.info(f"Output directory: {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)  # create dir f
    output_fit_t = output_dir / "gev_fit_temp.nc"
    output_fit = output_dir / "gev_fit.nc"
    output_summary = output_dir / "gev_summary.txt"
    output_fit_t_bs = output_dir / "gev_fit_temp_bs.nc"
    output_fit_bs = output_dir / "gev_fit_bs.nc"
    # deal with time_range argument. That modifies the output files.


    files = [output_fit, output_fit_t, output_summary]
    if args.bootstrap_samples > 0:
        my_logger.info(f"Bootstrapping")
        files.extend([output_fit_t_bs, output_fit_bs])
    # Modify the output files if time_range is set.
    if time_range is not None:
        files = [f.with_name(f"{f.stem}{out_post}{f.suffix}") for f in files]
        output_fit, output_fit_t, output_summary= tuple(files[0:3])
        if args.bootstrap_samples > 0:
            output_fit_t_bs, output_fit_bs = tuple(files[3:5])
    exist = [file.exists() for file in files]
    if all(exist) and (not args.overwrite):
        my_logger.warning(f"All output files {files} exist and overwrite not set. Exiting")
        sys.exit(0)
    vars_to_drop = ['xpos', 'ypos', 'fraction', 'sample_resolution', 'height', 'Observed_temperature']#, 'time','t']
    radar_dataset = xarray.open_dataset(args.input_file,chunks=dict(EventTime=-1,resample_prd=1,quantv=-1),cache=True,
                                        drop_variables=vars_to_drop)  # load the processed radar
    if time_range is not None:
        L=(radar_dataset.t >= time_range[0] )
        if time_range[1] is not None:
            L = L & (radar_dataset.t <= time_range[0])
        L = L.compute()
        radar_dataset = radar_dataset.where(L,drop=True)
    # remove nuisance variables
    radar_dataset = radar_dataset.drop_vars(['time','t'], errors='ignore') # TODO fix hardwiring so that this is not needed.


    if not use_dask:
        my_logger.debug('Loading data')
        radar_dataset=radar_dataset.load()
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
    if time_range is not None:
        extra_attrs.update(time_range=[str(t) for t in time_range])

    my_logger.info(f"Computing fits")

    fit, fit_bs = comp_radar_fit(radar_dataset,
                                 n_samples=args.nsamples, bootstrap_samples=args.bootstrap_samples,
                                 name='fit_nocov', file=output_fit, bootstrap_file=output_fit_bs,
                               extra_attrs=extra_attrs, recreate_fit=args.overwrite,use_dask=use_dask
                                 )
    my_logger.info(f"Computed no cov fits {ausLib.memory_use()}") # memory use

    fit_t, fit_t_bs = comp_radar_fit(radar_dataset, cov=['Tanom'],
                                     n_samples=args.nsamples, bootstrap_samples=args.bootstrap_samples,
                                     extra_attrs=extra_attrs, name='fit_temp', file=output_fit_t,
                                     bootstrap_file=output_fit_t_bs,
                                     recreate_fit=args.overwrite,use_dask=use_dask
                                     )
    my_logger.info(f"Computed time fits {ausLib.memory_use()}")


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
            q_aic_bs = (fit_t_bs.AIC - fit_bs.AIC).quantile([0.1, 0.5, 0.9], dim='bootstrap_sample').to_dataframe().unstack()
            print(f"AIC_bs: {q_aic_bs.round(-1)}", file=f)
        # Uncertainties from the covariance matrix
        cov_mean = fit_t.Cov.mean('sample')
        p_mean = fit_t.Parameters.mean('sample')
        ds_cov = xarray.Dataset(dict(cov=cov_mean, mean=p_mean))
        samp_params = ds_cov.groupby('resample_prd',squeeze=False).map(comp_dist,samples=args.nsamples).\
            sel(resample_prd=ds_cov.resample_prd)

        for p in ['location', 'scale']:
            dp = f'D{p}_Tanom'
            dfract = samp_params.sel(parameter=dp) / samp_params.sel(parameter=p)
            q = dfract.quantile([0.1, 0.5, 0.9], dim='sample').to_dataframe().unstack()
            print(f"   fract -cov  {dp}: {q.round(3)}", file=f)
            if args.bootstrap_samples > 0:
                dfract_bs = fit_t_bs.Parameters.sel(parameter=dp) / fit_t_bs.Parameters.sel(parameter=p)
                q_bs = dfract_bs.quantile([0.1, 0.5, 0.9], dim='bootstrap_sample').to_dataframe().unstack()
                print(f"BS fract -cov  {dp}: {q_bs.round(3)}", file=f)
    my_logger.info(f"Done")
