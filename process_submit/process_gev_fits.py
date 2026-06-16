#!/usr/bin/env python
# Compute the GEV fits -- uncertainties via sampling across events.
import argparse
import logging
import multiprocessing
import pathlib
import typing

import numpy as np
import pandas as pd
import scipy.stats
import xarray
from dask.array import random

import ausLib
import covariate

my_logger = logging.Logger('gev_fits')

filters=dict(
    Adelaide=lambda time: time.dt.year.isin([2015, 2016]),
    Wtakone = lambda time: time < np.datetime64("2014-12-01"),
    Melbourne = lambda time: time < np.datetime64('2003-12-01'),
) # fns applied to time co-ords in the events. Where True data will be removed


def do_gev_fit(dataset,
               cov: typing.Optional[typing.Union[list[str], str]] = None,
               recreate_fit: bool = False,
               file: typing.Optional[pathlib.Path] = None,
               name: typing.Optional[str] = None,
               extra_attrs: typing.Optional[dict] = None,
               use_dask: bool = True,
               initial_params: typing.Optional[xarray.Dataset] = None,
               dim: str = 'EventTime',
               **kwargs
               ) -> xarray.Dataset:
    if (file is not None) and file.exists() and (not recreate_fit):
        # got a file specified, it exists and we are not recreating fit
        data_set = xarray.load_dataset(file)  # just load the dataset and return it
        my_logger.info(f"Loaded existing data from {file}")
        return data_set

    vars_to_extract = ['max_value', 'count_cells']  # vars to extract from the dataset
    if cov is None:
        cov = []  # no covariates
    if isinstance(cov, str):
        cov = [cov]  # convert to a list,
    vars_to_extract.extend(cov)
    # normal stuff

    my_logger.info('Computing GEV fits')

    ds = dataset[vars_to_extract]
    mx = ds.max_value
    wt = ds.count_cells  # .broadcast_like(mx)
    if cov:  # got some covariances
        arrays = [xarray.ones_like(mx)]  # start with the intercept
        arrays += [ds[c] for c in cov]  # add on the covariates
        coord_names = ['intercept'] + ["D_" + c for c in cov]
        X_loc = xarray.concat(arrays, dim='coeff').transpose().assign_coords(coeff=coord_names)

    else:
        X_loc = None

    my_logger.debug('Doing GEV fit')

    fit = covariate.xarray_dist_fit(mx, distribution=scipy.stats.genextreme, dim=dim, X_loc=X_loc,
                                    use_dask=True, guess=initial_params, raise_error=False, **kwargs)  # )

    if use_dask:
        my_logger.debug('Computing for dask')
        logger = logging.getLogger('distributed.utils_perf')
        logger.setLevel('WARNING')
        fit = fit.compute()
        my_logger.debug('Done dask GEV computation')

    my_logger.debug('Done GEV fit')

    if name:
        fit.attrs.update(name=name)
    if extra_attrs:
        fit.attrs.update(extra_attrs)
    if file is not None:
        file.parent.mkdir(exist_ok=True, parents=True)  # make directory
        fit.to_netcdf(file)  # save the dataset.
        my_logger.info(f"Wrote fit information to {file}")
    return fit


def comp_radar_fit(
        dataset: xarray.Dataset,
        cov: typing.Optional[typing.Union[list[str], str]] = None,
        bootstrap_samples: int = 0,
        rng_seed: int = 123456,
        file: typing.Optional[pathlib.Path] = None,
        bootstrap_file: typing.Optional[pathlib.Path] = None,
        recreate_fit: bool = False,
        name: typing.Optional[str] = None,
        extra_attrs: typing.Optional[dict] = None,
        use_dask: bool = True,
        initial_params: typing.Optional[xarray.Dataset] = None,
        **kwargs,
) -> tuple[xarray.Dataset, typing.Optional[xarray.Dataset]]:
    """
    Compute the GEV fits for radar data.


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
    dim = 'EventTime'  # dimesnion over which we are collapsing (and bootstrapping)
    rng = random.default_rng(rng_seed)
    fit = do_gev_fit(dataset, cov=cov,
                     recreate_fit=recreate_fit, file=file, name=name, dim=dim,
                     extra_attrs=extra_attrs, use_dask=use_dask, initial_params=initial_params, **kwargs)

    if bootstrap_samples == 0:
        return fit, None  # no bootstrapping wanted.

    if (bootstrap_file is not None) and bootstrap_file.exists() and (
            not recreate_fit):  # got a file specified, it exists and we are not recreating fit
        my_logger.debug(f'Loading bootstrap fit from {bootstrap_file}')
        bs_fit = xarray.load_dataset(bootstrap_file)  # just load the dataset
        return fit, bs_fit  # can now return the fit and bootstrap fit

    else:  # need to do the bootstrap fit.

        my_logger.info(f"Calculating bootstrap for {name} with cov {cov} {ausLib.memory_use()}")
        # generate all the samples and hope that dask saves the day :-)
        init_params = xarray.Dataset(
            dict(c=fit.shape, loc_coefficients=fit.location, mu_coefficients=fit.mu))  # have some decent initial guess.
        ## AI guided below.ob bootstrapping though it missed the drop_vars...
        N = dataset[dim].size
        idx = rng.integers(0, N, size=(bootstrap_samples, N), chunks=(1, -1))
        idx_da = xarray.DataArray(idx, dims=('bootstrap_sample', 'sample')).compute()
        ds_boot = dataset.isel({dim: idx_da}).drop_vars(dim)
        fit_boot = do_gev_fit(ds_boot, cov=cov,
                              recreate_fit=recreate_fit, file=bootstrap_file, name=name, dim='sample',
                              extra_attrs=extra_attrs, use_dask=use_dask, initial_params=init_params,
                              **kwargs)
        if extra_attrs:
            fit_boot.attrs.update(extra_attrs)
        if bootstrap_file:
            fit_boot.to_netcdf(bootstrap_file)

    return fit, fit_boot



def comp_summ_stats(fit:xarray.Dataset,
                    fit_cov:xarray.Dataset,
                    output_file:pathlib.Path, cov_var: str,
                    sample_dim: typing.Optional[str] = None, ):
    quants = [0.1, 0.5, 0.9]
    coords_to_drop = [c for c in fit.coords if c not in fit.dims]
    with output_file.open('wt') as f:
        print(f"Summary of GEV fits for {cov_var}", file=f)
        qv_sel = dict(quantv=0.5, method='nearest', tolerance=0.01)

        aic = fit_cov.AIC.sel(**qv_sel).drop_vars(coords_to_drop,errors='warn').squeeze(drop=True).drop_vars('quantv',errors='warn')
        if sample_dim:
            aic = aic.quantile([0.1, 0.5, 0.95], dim=sample_dim)
        aic = aic.to_dataframe().unstack()
        print(f"{aic.round(-1)}", file=f)


        delta_aic = (fit_cov.AIC - fit.AIC).sel(**qv_sel).drop_vars(coords_to_drop,errors='warn').squeeze(drop=True).drop_vars('quantv',errors='warn')
        if sample_dim:
            delta_aic = delta_aic.quantile([0.1, 0.5, 0.95], dim=sample_dim)
        delta_aic = delta_aic.to_dataframe().unstack()
        print(f"Delta {delta_aic.round(-1)}", file=f)
        dfract_loc = fit_cov.location.isel(loc_coeff=1) / fit_cov.location.isel(loc_coeff=0)
        dscale_loc = np.exp(fit_cov.mu.isel(mu_coeff=1))
        coords_to_drop+=['loc_coeff','mu_coeff','quantv']
        for var, p in zip([dfract_loc, dscale_loc], ['location', 'scale']):
            if sample_dim:
                var_qs = var.sel(**qv_sel).quantile(quants, dim=sample_dim)
            else:
                var_qs = var.sel(**qv_sel)
            var_qs = var_qs.drop_vars(coords_to_drop, errors='warn').to_dataframe().unstack()
            print(f"   fract cov  {p}: {var_qs.round(3)}", file=f)
    my_logger.debug(f'Wrote summary file {output_file}')


if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!
    parser = argparse.ArgumentParser(description="Compute GEV fits for radar event  data")
    parser.add_argument('input_file', type=pathlib.Path, help='input file of events from  radar data')
    parser.add_argument('--outdir', type=pathlib.Path,
                        help='output directory for fits. If not provided - computed from input file name'
                        )
    parser.add_argument('--bootstrap_samples', type=int, help='Number of event bootstraps to use', default=0)
    parser.add_argument('--covariates', nargs='+', default=['temperature'], help='List of covariates to use')
    parser.add_argument('--penalty_infinite_pdf',type=float,default=None,
                        help='Penalty for infinities in fitting')
    parser.add_argument('--penalty_outside_support',type=float,default=None,
                        help='Penalty for being outside support in fitting')
    # Names of covariate to use.
    ausLib.add_std_arguments(parser)  # add on the std args
    args = parser.parse_args()
    if args.outdir is None:
        parent = args.input_file.parent
        output_dir = parent / 'gev_fits_new'
    else:
        output_dir = args.outdir

    use_dask = args.dask



        ## defaults for submission.
    pbs_log_dir = output_dir / 'pbs_logs'
    run_log_dir = output_dir / 'run_logs'
    log_file = 'gev_fits' + pd.Timestamp.now('UTC').strftime('%Y_%m_%d_%H%M%S')
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
    files = dict(fit=output_dir / 'gev_fit.nc',
                 summary=output_dir / 'gev_summary.txt')  # std fit.

    for cov in args.covariates:  # add on the files generated for covariates
        files.update({f'fit_{cov}': output_dir / f'gev_fit_{cov}.nc'})  # covariate fit.
        files.update({f'summary_{cov}': output_dir / f'gev_summary_{cov}.txt'})  # summary fit.
    # deal with bootstrap files.
    if args.bootstrap_samples > 0:
        files.update(dict(fit_bs=output_dir / f'gev_fit_bs.nc', summary_bs=output_dir / f'gev_summary_bs.txt'))
        for cov in args.covariates:
            files.update({
                f'fit_{cov}_bs': output_dir / f'gev_fit_{cov}_bs.nc',  # fit file for bs
                f'summary_{cov}_bs': output_dir / f'gev_summary_{cov}_bs.txt'  # summary file for bs
            })



    my_logger = ausLib.process_std_arguments(args, default=default, files=list(files.values()))  # setup the std stuff
    my_logger.debug(f'use_dask is {use_dask}')
    my_logger.info(f"Output directory: {output_dir}")
    if args.bootstrap_samples > 0:
        my_logger.info(f"Bootstrapping")
    my_logger.info(f'Output files: {files.values()}')
    # work out what is needed from the event file
    vars_to_extract = ['max_value', 'count_cells', 't']  # vars to extract from the dataset
    vars_to_extract.extend(args.covariates)  # plus the covariates
    radar_dataset = xarray.open_dataset(args.input_file,
                                        chunks=dict(EventTime=-1, resample_prd=-1, quantv=4)) # load radar events
    radar_dataset = radar_dataset[vars_to_extract].load()  # load the processed radar
    site = radar_dataset.attrs['site'] # want an error if site not found
    filter_fn = filters.get(site)  # get the filter function for the site (if it exists)
    if filter_fn is not None:
        mask = filter_fn(radar_dataset.t)

        my_logger.info(f"Site {site} filtered out {int(mask.sum())} events from {int(np.isfinite(radar_dataset.t).sum())} events")
        radar_dataset = radar_dataset.where(~mask) # filter




    threshold = 0.5  # some max are zero -- presumably no rain then.
    msk = (radar_dataset.max_value > threshold)
    for v in radar_dataset.data_vars:
        if set(msk.dims) == set(radar_dataset[v].dims):  # only do this where variable dims match msk dims
            radar_dataset[v] = radar_dataset[v].where(msk, drop=True)
    # set up metadata.
    extra_attrs = radar_dataset.attrs.copy()
    extra_attrs.update(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.now('UTC').isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()]
                       )

    # do first computation with no covariate.
    my_logger.info(f"Computing fits")


    fit, fit_bs = comp_radar_fit(radar_dataset,
                                 bootstrap_samples=args.bootstrap_samples,
                                 name='fit_nocov', file=files['fit'], bootstrap_file=files.get('fit_bs'),
                                 extra_attrs=extra_attrs, recreate_fit=args.overwrite, use_dask=use_dask,
                                 penalty_infinite_pdf=args.penalty_infinite_pdf,
                                 penalty_outside_support=args.penalty_outside_support,

                                 )

    my_logger.info(f"Computed no cov fits {ausLib.memory_use()}")  # memory use
    ## loop over the covariates and do the fits.
    for cov_var in args.covariates:
        my_logger.info('Doing covariate fit for ' + cov_var)
        mn_value = float(radar_dataset[cov_var].mean())
        radar_dataset[cov_var + '_anom'] = radar_dataset[
                                              cov_var] - mn_value  # make the anomaly and add it to the dataset.
        extra_attrs_cov = extra_attrs.copy()
        extra_attrs_cov.update({f'mean_{cov_var}': mn_value})

        fit_cov, fit_cov_bs = comp_radar_fit(radar_dataset, cov=[cov_var + '_anom'],
                                             bootstrap_samples=args.bootstrap_samples,
                                             extra_attrs=extra_attrs_cov, name=f'fit_{cov_var}',
                                             file=files[f'fit_{cov_var}'],
                                             bootstrap_file=files.get(f'fit_{cov_var}_bs'),
                                             recreate_fit=args.overwrite,
                                             use_dask=use_dask,
                                             penalty_infinite_pdf=args.penalty_infinite_pdf,
                                             penalty_outside_support=args.penalty_outside_support,
                                             )
        my_logger.info(f"Computed fits for {cov_var} {ausLib.memory_use()}")
        ## write out summary info


        comp_summ_stats(fit, fit_cov, files[f'summary_{cov_var}'], cov_var)
        my_logger.info(f"Computed summary for {cov_var} {ausLib.memory_use()}")

        if args.bootstrap_samples > 0: # write out the bs summary info if we have done the bs.
            comp_summ_stats(fit_bs, fit_cov_bs, files[f'summary_{cov_var}_bs'], cov_var, sample_dim='bootstrap_sample')

    my_logger.info(f"Done")
