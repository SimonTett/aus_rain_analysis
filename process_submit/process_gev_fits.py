#!/usr/bin/env python
# Compute the GEV fits -- uncertainties via sampling across events.
import logging
import multiprocessing
import pathlib

import typing

import numpy as np
import scipy.stats
import xarray
import ausLib

import argparse
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
        rng = np.random.default_rng(123456)
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

def do_gev_fit(dataset,
               cov:typing.Optional[typing.Union[list[str],str]]=None,
               rng:typing.Optional[np.random.Generator]=None,
               n_samples:int=100,
               verbose:bool=False,
               recreate_fit:bool=False,
               file:typing.Optional[pathlib.Path]=None,
               name:typing.Optional[str]=None,
               extra_attrs:typing.Optional[dict]=None,
               use_dask:bool=False,
               initial_params:typing.Optional[xarray.DataArray]=None
               ) -> xarray.Dataset:
    from R_python import gev_r # do this here as close to when we want it
    if rng is None:
        rng = np.random.default_rng(123456) # setup the random nuber generatpr oif not already provided

    vars_to_extract = ['max_value', 'count_cells']  # vars to extract from the dataset
    if cov is  None:
        cov = []  # no covariates
    if isinstance(cov, str):
        cov = [cov] # convert to a list,
    vars_to_extract.extend(cov)
    # normal stuff

    my_logger.info('Computing GEV fits')

    rand_index = rng.integers(1, len(dataset.quantv) - 2, size=n_samples)  # don't want the min or max quantiles
    coord = dict(sample=np.arange(0, n_samples))
    # randomly select from quantiles for each event -- they all exist (or not)
    ds = dataset[vars_to_extract].isel(quantv=rand_index). \
        rename(dict(quantv='sample')).assign_coords(**coord)

    if use_dask:
        my_logger.debug(f'Chunking data for best est  {ausLib.memory_use()}')
        samp_chunk = max(n_samples // 20, 1)
        samp_chunk = 4
        ds = ds.chunk(sample=samp_chunk, resample_prd=-1, EventTime=-1)  # parallelize over sample
        my_logger.debug(f'Chunk size is: {ds.chunksizes}')
        my_logger.debug(f'Rechunked data for best est  {ausLib.memory_use()}')

    cov_rand = [ds[c] for c in cov]

    wt = ds.count_cells
    mx = ds.max_value
    my_logger.debug('Doing GEV fit')
    fit = gev_r.xarray_gev(mx, cov=cov_rand, dim='EventTime', weights=wt, verbose=verbose,
                           recreate_fit=recreate_fit, file=file, name=name, extra_attrs=extra_attrs,
                           use_dask=use_dask, initial=initial_params)
    my_logger.debug('Done GEV fit')
    return fit

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
        verbose: bool = False,
        initial_params: typing.Optional[xarray.DataArray] = None,
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
        initial_params: initial parameters for the fit.

    Returns:fit

    """
    # do fit on data.
    rng = random.default_rng(rng_seed)
    fit = do_gev_fit(dataset, cov=cov, rng=rng, n_samples=n_samples,
                     verbose=verbose, recreate_fit=recreate_fit, file=file, name=name,
                     extra_attrs=extra_attrs, use_dask=use_dask, initial_params=initial_params)



    if  bootstrap_samples == 0:
        return fit,None # no bootstrapping wanted.

    if (bootstrap_file is not None) and bootstrap_file.exists() and (
            not recreate_fit):  # got a file specified, it exists and we are not recreating fit
        my_logger.debug(f'Loading bootstrap fit from {bootstrap_file}')
        bs_fit = xarray.load_dataset(bootstrap_file).mean('sample')  # just load the dataset
        return fit, bs_fit  # can now return the fit and bootstrap fit

    # need to compute the bootstrap fit.
    my_logger.info(f"Calculating bootstrap for {name} {ausLib.memory_use()}")
    # loop over bootstrap samples and then concat together at the end.
    # generating all the samples produces a very large mem object and requires data
    # to be moved around. So don't bother...
    bs_sample_fits = []  # where the individual samples go.
    init_params = fit.median('sample').Parameters
    # get the initial parameters. Speeds up the fit by about 10-20% and hopefully reduces odds of bad fits...
    rng = random.default_rng(rng_seed) # same seed as used in the gev fit. (which will have its own rng)
    for samp in range(bootstrap_samples):
        my_logger.debug('Bootstrapping')
        # sample the events with replacement.

        ds_bs = dataset.groupby('resample_prd',squeeze=False).map(sample_events2, rng=rng,
                                               dim='EventTime'). \
            drop_vars('EventTime').assign_coords(bootstrap_sample=samp)
        bs_fit = do_gev_fit(ds_bs, cov=cov, rng=rng, n_samples=n_samples,
                     verbose=verbose, name=name,use_dask=use_dask, initial_params=initial_params)
        bs_sample_fits.append(bs_fit)
        my_logger.info(f'Done BS fit for sample {samp} {ausLib.memory_use()}')
        # all done doing the individual fits. Now need to concat them together.
    bs_fit = xarray.concat(bs_sample_fits, dim='bootstrap_sample')

    if extra_attrs:
        bs_fit.attrs.update(extra_attrs)
    if bootstrap_file:
        bs_fit.to_netcdf(bootstrap_file)
        bs_fit = bs_fit.mean('sample')

    return fit, bs_fit

def comp_dist(ds, samples=100):

    dist = scipy.stats.multivariate_normal(ds['mean'].squeeze(),
                                           ds['cov'].squeeze(),
                                           allow_singular=True
                                           )
    dist_sample = dist.rvs(size=samples)
    dist_sample = xarray.DataArray(data=dist_sample,
                                   coords=dict(sample=np.arange(0, args.nsamples), parameter=ds['mean'].parameter)
                                   )
    dist_sample = dist_sample.rename('cov_param_samp')
    return dist_sample


def comp_summ_stats(fit, fit_cov, output_file, nsamples, cov_var, sample_dim):
    with output_file.open('wt') as f:
        # print out the AIC
        q_aic = (fit_cov.AIC - fit.AIC).quantile([0.1, 0.5, 0.9], dim=sample_dim).to_dataframe().unstack()
        print(f"AIC: {q_aic.round(-1)}", file=f)
        # Uncertainties from the covariance matrix
        cov_mean = fit_cov.Cov.mean('sample', skipna=True)
        p_mean = fit_cov.Parameters.mean('sample', skipna=True)
        if cov_mean.isnull().any():
            raise ValueError('Covariance matrix has NaNs')
        if p_mean.isnull().any():
            raise ValueError('Parameters matrix has NaNs')

        ds_cov = xarray.Dataset(dict(cov=cov_mean, mean=p_mean))
        samp_params = ds_cov.groupby('resample_prd', squeeze=False). \
            map(comp_dist, samples=nsamples).sel(resample_prd=ds_cov.resample_prd)

        for p in ['location', 'scale']:
            dp = f'D{p}_{cov_var}anom'
            dfract = samp_params.sel(parameter=dp) / samp_params.sel(parameter=p)
            q = dfract.quantile([0.1, 0.5, 0.9], dim=sample_dim).to_dataframe().unstack()
            print(f"   fract cov  {dp}: {q.round(3)}", file=f)
    my_logger.debug(f'Wrote summary file {output_file}')
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
    parser.add_argument('--covariates', nargs='+',  default=['temperature'],help='List of covariates to use')
    # Names of covariate to use.
    ausLib.add_std_arguments(parser)  # add on the std args
    args = parser.parse_args()
    if args.outdir is None:
        parent = args.input_file.parent
        output_dir = parent / 'fits'
    else:
        output_dir = args.outdir
        # base_dir = output_dir



    use_dask = args.dask
    if use_dask:
        raise NotImplementedError("dask  and rpy2 do not work well together")
    # check times are 1 or 2 element. If 1 element then assume it is the start time.
    time_range = None
    if args.time_range is not None:
        if len(args.time_range) == 1:
            time_range = (args.time_range[0], None)
            out_post = args.time_range[0].strftime('_%Y%m%d')
        elif len(args.time_range) == 2:
            time_range = tuple(args.time_range)
            out_post = "_" + "_".join([tr.strftime('%Y%m%d') for tr in args.time_range])
        else:
            parser.print_help()
            raise ValueError("Time range must be 1 or 2 elements")

        ## defaults for submission.
    pbs_log_dir = output_dir / 'pbs_logs'
    run_log_dir = output_dir / 'run_logs'
    log_file = 'gev_fits' + pd.Timestamp.utcnow().strftime('%Y_%m_%d_%H%M%S')
    log_base_default = pbs_log_dir / log_file
    log_file_default = run_log_dir / f'{log_file}.log'
    job_name_default = f'gev_{output_dir.stem[0:10]}'
    json_file_default = ausLib.module_path / 'config_files/process_gev_fits.json'

    default = dict(log_base=log_base_default,  # where pbs logs (stdout & stderr)
                   log_file=log_file_default,  # log while run is going
                   job_name=job_name_default,  # default name of job
                   json_submit_file=json_file_default)

    output_dir.mkdir(exist_ok=True, parents=True)  # create dir f
    # set up list of output files
    files = dict(fit= output_dir/'gev_fit.nc',
             summary=output_dir/'gev_summary.txt')  # std fit.

    for cov in args.covariates: # add on the files generated for covariates
        files.update({f'fit_{cov}':output_dir/f'gev_fit_{cov}.nc'}) # covariate fit.
        files.update({f'summary_{cov}':output_dir/f'gev_summary_{cov}.txt'})  # summary fit.
    # deal with bootstrap files.
    if args.bootstrap_samples > 0:
        files.update(dict(fit_bs=output_dir/f'gev_fit_bs.nc', summary_bs=output_dir/f'gev_summary_bs.txt'))
        for cov in args.covariates:
            files.update({
                f'fit_{cov}_bs':output_dir/f'gev_fit_{cov}_bs.nc', # fit file for bs
                f'summary_{cov}_bs':output_dir/f'gev_summary_{cov}_bs.txt' # summary file for bs
                })


    # Modify the output files if time_range is set.
    if time_range is not None: # filenames change if have time range
        files = {k:f.with_name(f"{f.stem}{out_post}{f.suffix}") for k,f in files.items()}

    my_logger = ausLib.process_std_arguments(args, default=default,files=list(files.values()))  # setup the std stuff
    my_logger.debug(f'use_dask is {use_dask}')
    my_logger.info(f"Output directory: {output_dir}")
    if args.bootstrap_samples > 0:
        my_logger.info(f"Bootstrapping")
    my_logger.info(f'Output files: {files.values()}')
    # work out what is needed from the event file
    vars_to_extract = ['max_value', 'count_cells','t']  # vars to extract from the dataset
    vars_to_extract.extend(args.covariates) # plus the covariates
    radar_dataset = xarray.open_dataset(args.input_file, chunks=dict(EventTime=-1, resample_prd=1, quantv=-1),
                                        cache=True)[vars_to_extract]  # load the processed radar
    if time_range is not None:
        L:xarray.DataArray = (radar_dataset.t >= time_range[0])
        if time_range[1] is not None:
            L = L & (radar_dataset.t <= time_range[1])
        L = L.load()
        radar_dataset = radar_dataset.where(L, drop=True)

    if not use_dask:
        my_logger.debug('Loading data')
        radar_dataset = radar_dataset.load()

    threshold = 0.5  # some max are zero -- presumably no rain then.
    msk = (radar_dataset.max_value > threshold)
    for v in radar_dataset.data_vars:
        if set(msk.dims) == set(radar_dataset[v].dims): # only do this where variable dims match msk dims
            radar_dataset[v] = radar_dataset[v].where(msk, drop=True)
    # set up metadata.
    extra_attrs = radar_dataset.attrs.copy()
    extra_attrs.update(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.utcnow().isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()]
                       )
    if time_range is not None:
        extra_attrs.update(time_range=[str(t) for t in time_range])
    # do first computation with no covariate.
    my_logger.info(f"Computing fits")
    dd = radar_dataset.max_value.sel(quantv=0.5, drop=True)
    initial_params = [dd.mean('EventTime'), dd.std('EventTime')]
    initial_params = xarray.concat(initial_params, dim='parameter', coords='minimal').assign_coords(
        parameter=['location', 'scale'])
    # best not to provide initial guess for shape as can cause problems...
    # shape = xarray.DataArray(0.1).broadcast_like(initial_params[0]).assign_coords(parameter='shape')
    # initial_params = xarray.concat([initial_params, shape], dim='parameter',coords='minimal')

    fit, fit_bs = comp_radar_fit(radar_dataset,
                                 n_samples=args.nsamples, bootstrap_samples=args.bootstrap_samples,
                                 name='fit_nocov', file=files['fit'], bootstrap_file=files.get('fit_bs'),
                                 extra_attrs=extra_attrs, recreate_fit=args.overwrite, use_dask=use_dask,
                                 verbose=args.verbose > 2,
                                 initial_params=initial_params
                                 )
    my_logger.info(f"Computed no cov fits {ausLib.memory_use()}")  # memory use
    ## loop over the covariates and do the fits.
    for cov_var in args.covariates:
        my_logger.info('Doing covariate fit for ' + cov_var)
        mn_value = float(radar_dataset[cov_var].mean())
        radar_dataset[cov_var + 'anom'] = radar_dataset[cov_var] - mn_value  # make the anomaly and add it to the dataset.
        extra_attrs = radar_dataset.attrs.copy()
        extra_attrs.update(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.utcnow().isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()]
                       )
        extra_attrs.update({f'mean_{cov_var}': mn_value})



        fit_cov, fit_cov_bs = comp_radar_fit(radar_dataset, cov=[cov_var + 'anom'],
                                         n_samples=args.nsamples, bootstrap_samples=args.bootstrap_samples,
                                         extra_attrs=extra_attrs, name=f'fit_{cov_var}',
                                         file=files[f'fit_{cov_var}'],
                                         bootstrap_file=files.get(f'fit_{cov_var}_bs'),
                                         recreate_fit=args.overwrite,
                                         use_dask=use_dask,
                                         verbose=args.verbose > 2,
                                         initial_params=initial_params # using the same initial params as no covariate.
                                         )
        my_logger.info(f"Computed time fits {ausLib.memory_use()}")
        ## write out summary info

        comp_summ_stats(fit,fit_cov,files[f'summary_{cov_var}'],args.nsamples,cov_var,'sample')
        my_logger.info(f"Computed summary for {cov_var} {ausLib.memory_use()}")

        if args.bootstrap_samples > 0:
            comp_summ_stats(fit_bs,fit_cov_bs,files[f'summary_{cov_var}_bs'],args.nsamples,cov_var,'bootstrap_sample')

    my_logger.info(f"Done")

