#!/usr/bin/env python
# process reflectivity data
# currently y and x_bounds are nan. Probably coming from when they get merged in,
# and merge seems to be doing something odd to the co-ords...
import pathlib
import typing
import argparse
import numpy as np
import xarray
import multiprocessing
import ausLib
from ausLib import memory_use, my_logger, site_numbers
import dask


def empty_ds(example_da: xarray.DataArray, resample_prd: typing.List[str],
             non_time_variables: typing.Union[typing.List[str], str],
             vars_time: typing.Union[typing.List[str], str],
             no_resample_vars: typing.Optional[typing.Union[typing.List[str], str]] = None,
             ) -> xarray.Dataset:
    """
    Return an empty dataset
    :param resample_prd: list of resample periods
    :param example_da:
    :param non_time_variables:
    :param vars_time:
    :param no_resample_vars -- list of variables that should not have a dimension resample_prd
    :return: empty dataset.
    """
    # deal with singleton vars
    if isinstance(non_time_variables, str):
        non_time_variables = [non_time_variables]
    if isinstance(vars_time, str):
        vars_time = [vars_time]
    if isinstance(no_resample_vars, str):
        no_resample_vars = [no_resample_vars]

    if no_resample_vars is None:
        no_resample_vars = []

    # generates vars
    result = dict()
    for var in non_time_variables:
        result[var] = example_da.where(False).rename(var)
    for var in vars_time:
        result[var] = example_da.where(False).rename(var).astype('<M8[ns]')
        result[var].attrs.pop('units', None)  # remove the units here.
    # add resample_prd as dim on
    for key in result.keys():
        if key not in no_resample_vars:
            result[key] = result[key].expand_dims(resample_prd=resample_prd)
    result = xarray.Dataset(result)
    return result

## summary_process def
def summary_process(data_array: xarray.DataArray,
                    mean_resample: typing.Union[typing.List[str], str] = None,
                    time_dim: str = 'time',
                    threshold: typing.Optional[float] = None,
                    base_name: typing.Optional[str] = None) -> typing.Optional[xarray.Dataset]:
    f"""
    Process data_array for max (and mean & time of max).

    Args:
        data_array: dataArray to be processed. 
        mean_resample: the resample period(s) to generate means.
        base_name: basename to be used. If not provided then long_name in attrs of data_array will be used. 
        time_dim: the name of the time dimension
        threshold: the threshold above which var is used in *_thresh dataArrays. 
        If not provided All _thresh vars will be empty


    Returns:dataset containing summary values. These are:
         max_"base_name", time_max_"base_name",  mean_"base_name", median_"base_name",
         max, time of max, mean, median for values from resampled values. 
          median_"base_name"_thresh median for values > rain_threshold.
          mean_rate_thresh mean for values > rain_thresh
    Also includes some count values.
            f'count_raw_{base_name}_thresh: number of values above threshold before resampling
            f'count_raw_{base_name}: number of raw values in time period.
            f'count_{base_name}': count of resampled samples  
            f'count_max_{base_name}': max number of samples in resample. 


    """

    time_bounds = [data_array.time.min().values, data_array.time.max().values]
    time_bounds = xarray.DataArray(time_bounds, dims='bounds').rename('time_bounds')
    time_str = f"{time_bounds.values[0]} - {time_bounds.values[1]}"
    if mean_resample is None:
        mean_resample = ['1h']
    if isinstance(mean_resample, str):
        mean_resample = [mean_resample]
    if base_name is None:
        base_name = data_array.attrs.get('long_name')
        if base_name is None:  # raise an error
            raise ValueError('base_name not provided and not in dat_array.attrs["long_name"]')

    # set up empty result
    vars_to_gen = [f"max_{base_name}", f"median_{base_name}", f"mean_raw_{base_name}",
                   f'mean_{base_name}', f'count_raw_{base_name}',f'count_raw_miss_{base_name}']
    if threshold is not None:  # add on threshold vars
        vars_to_gen += [f"median_{base_name}_thresh", f"count_raw_{base_name}_thresh",
                        f"mean_{base_name}_thresh", f"count_{base_name}_thresh"]
    # lit of variables that won't have resample_prd included.
    no_resample_vars = [var for var in vars_to_gen if 'raw' in var]
    no_resample_vars += ['time_bounds', 'sample_resolution']

    result = empty_ds(data_array.isel({time_dim: 0}, drop=True), mean_resample,
                      vars_to_gen, f"time_max_{base_name}",
                      no_resample_vars=no_resample_vars)
    # add in counts of samples
    dummy = xarray.DataArray(data=np.repeat(np.nan, len(mean_resample)),
                             coords=dict(resample_prd=mean_resample))
    for n in ['count_max', 'count']:
        result[f'{n}_{base_name}'] = dummy
    # and time_bounds
    result['time_bounds'] = time_bounds
    count_raw = data_array[time_dim].count()
    result[f'count_raw_{base_name}'] = count_raw
    result[f'count_raw_miss_{base_name}'] = data_array.isnull().sum(time_dim)
    # add in time resolution and complain (but continue) if have values diff from median
    td = data_array.time.diff(time_dim)
    time_resoln = td.median().values
    result['sample_resolution'] = time_resoln
    L = td != time_resoln
    if L.any():
        scale = np.timedelta64(1, 'm')
        bad_td = td.where(L, drop=True) / scale
        my_logger.warning(
            f"Time resolution not consistent -- median {time_resoln / scale} mins with {len(bad_td)} bad values {np.unique(bad_td.values)}")
        for v in bad_td:
            my_logger.debug(f'{v.values} mins @ {v.time.values}')

    # check we have some data and if not exit!
    if int(count_raw) == 0:  # no values
        my_logger.warning(f"No data for {time_str}")
        return result
    bad = data_array.isnull().all(time_dim, keep_attrs=True)  # find out where *all* data null
    if bad.all():
        my_logger.error(f"All missing at {time_str}")
        return result

    my_logger.debug(f"Processing data for {time_str} {memory_use()}")
    # get on with processing data now!

    # check all values are finite.
    if np.isinf(data_array).any():
        raise ValueError('Inf values in data_array')
    if threshold:  # compute threshold vars.
        my_logger.debug('Computing threshold raw count')
        var_thresh_count = (data_array > threshold).sum(time_dim, skipna=True, min_count=1, keep_attrs=True)
        var_thresh_count.attrs.update(long_name=f'# times  >{threshold}')
        result[f'count_raw_{base_name}_thresh'] = var_thresh_count
    var_mean = data_array.mean(time_dim, skipna=True, keep_attrs=True)
    result[f'mean_raw_{base_name}'] = var_mean
    list_resamp_result = []  # list of datasets for each resample prd.
    for mn_resamp in mean_resample:
        rr = result.sel(resample_prd=mn_resamp).drop_vars(
            no_resample_vars)  # remove those vars that don't have resampling!
        mean_resamp = (data_array.resample({time_dim: mn_resamp}).mean(skipna=True).
                       expand_dims(resample_prd=[mn_resamp]))
        count_resamp = (data_array[time_dim].resample({time_dim: mn_resamp}).count().
                        expand_dims(resample_prd=[mn_resamp])).assign_attrs(extra='Maximum count in resample period')
        max_samps = count_resamp.max()
        ok = (count_resamp == max_samps)
        mean_resamp = mean_resamp.where(ok, drop=True)  # mask where have complete data
        # for max want to have consistent no of values.
        if np.isinf(mean_resamp).any():
            raise ValueError(f'Inf in mean_resamp')
        my_logger.debug(
            f"mean and count computed using {mn_resamp} for {time_str} {memory_use()}")

        # do the actual computation for this resampling period.
        if threshold:  # compute threshold vars.
            my_logger.debug(f'Computing thresholded variables for {mn_resamp}')
            m = mean_resamp > threshold  # mask for  threshold.
            mean_resamp_thresh = mean_resamp.where(m)
            count_thresh = m.sum(time_dim)
            count_thresh.attrs['long_name'] = f'Count of {base_name} {mean_resample} > {threshold} '
            median_thresh = mean_resamp_thresh.median(time_dim, keep_attrs=True, skipna=True)
            mean_thresh = mean_resamp_thresh.mean(time_dim, keep_attrs=True, skipna=True)

            rr.update({
                f"median_{base_name}_thresh": median_thresh,
                f"mean_{base_name}_thresh": mean_thresh,
                f"count_{base_name}_thresh": count_thresh
            })
        my_logger.debug(f'Computed threshold vars for {mn_resamp}')
        max_mean_resamp = mean_resamp.max(time_dim, keep_attrs=True, skipna=True)
        time_max_resamp = mean_resamp.idxmax(time_dim, keep_attrs=False, skipna=True)
        median_mean_resamp = mean_resamp.median(time_dim, keep_attrs=False, skipna=True)
        mean_mean_resamp = mean_resamp.mean(time_dim, keep_attrs=False, skipna=True)
        count_max_samples = ok.sum(time_dim, keep_attrs=False)
        count_max_samples.attrs.update(long_name='Count_samples')
        rr.update({
            f"max_{base_name}": max_mean_resamp,
            f"median_{base_name}": median_mean_resamp,
            f"mean_{base_name}": mean_mean_resamp,
            f"time_max_{base_name}": time_max_resamp,
            f'count_{base_name}': count_max_samples,
            f'count_max_{base_name}': max_samps
        })
        # check everything is finite
        bad_vars = []
        for key, var in rr.items():
            if np.isinf(var).any():
                bad_vars.append(key)
                my_logger.warning(f'Inf values in {key}')
        if len(bad_vars) > 0:
            raise ValueError(f'Inf values in {bad_vars}')
        list_resamp_result.append(rr)
    # end of looping over time resample periods
    rr = xarray.concat(list_resamp_result, dim='resample_prd')
    result.update(rr)


    # sort out meta-data.
    variables = [v for v in result.variables]
    for k in variables:
        if k.startswith("time_"):
            comp = k.split('_')[1]
            result[k].attrs['long_name'] = 'time of ' + comp + " " + base_name
            result[k].encoding.update(units='minutes since 1990-01-01T00:00')
        elif k.endswith("thresh"):
            comp = k.split('_')[0]
            result[k].attrs['long_name'] = comp + " " + base_name + f'  > {threshold}'

        else:
            comp = k.split('_')[0]
            result[k].attrs['long_name'] = comp + " " + base_name

    result['threshold'] = threshold
    # add on time.
    time = data_array[time_dim].mean()  # mean time
    time.attrs.update(data_array[time_dim].attrs)
    time.attrs['long_name'] = 'Time'
    result = result.expand_dims({time_dim: [time.values]})
    result.attrs.update(data_array.attrs)  # preserve initial metadata.
    my_logger.info(f"Summaries computed for  {time_str} {memory_use()}")
    return result

##
def fix_spatial_units(ds: xarray.Dataset):
    """
    Fix up the spatial units in the dataset
     (x,y,x_bounds,y_bounds) are converted from km to m.
    :param ds: dataset for which x etc are to b converted. Replacement done in place.
    :return: Nada
    """
    for c in ['x', 'y', 'x_bounds', 'y_bounds']:  # convert all co-ords to m from km,
        if c not in ds.variables:  # don't have the variable so skip further processing
            continue
        try:
            unit = ds[c].attrs['units']
        except KeyError:  # no units set
            unit = 'km'
            my_logger.debug(f'Set units on  {c} to  km')
        try:
            if unit == 'km':  # convert to m
                with xarray.set_options(keep_attrs=True):
                    ds[c] = ds[c] * 1000.0  # convert to meters
                ds[c] = ds[c].assign_attrs(units='m')
                my_logger.debug(f'Converted {c} to meters from km')
        except KeyError:
            pass

    return ds


##
type_cm = typing.Literal['mean', 'median']


def read_zip(path: pathlib.Path,
             concat_dim: str = 'valid_time',
             coarsen: typing.Optional[typing.Dict[str, int]] = None,
             coarsen_method: type_cm = 'mean',
             bounds_vars: typing.Tuple[str] = ('y_bounds', 'x_bounds'),
             dbz_ref_limits: typing.Optional[typing.Tuple[float, float]] = None,
             coarsen_cv_max: typing.Optional[float] = None,
             check_finite: bool = True,
             first_file: bool = False,
             **load_kwargs
             ) -> typing.Optional[xarray.Dataset]:
    """
    Read radar zip file


    :param first_file: Passed to ausLib.read_radar_zipfile
    :param check_finite:  Check data is finite (not inf) -- see ausLib.coarsen_ds
    :param path: path to zip file to be read in
    :param concat_dim: Dimension over which to concatenate

    :param dbz_ref_limits: tuple of limits for reflectivity in dbz.
       Values below lower limit are set to 0.0 when converted to reflectivty and above to missing.
        If None no limits are applied.
    :param coarsen: dict for coarsening. If None no coarsening done
    :param coarsen_method: method to coarsen by
    :param coarsen_cv_max: Maximum coefficient of variability (std/mean) for coarsened data.Values above this are set to missing.  If set var_speckle will be computed
    :param bounds_vars:  tuple of variables that are bounds
    :param load_kwargs: kwargs for loading in read_radar_zipfile
    :return:
    """

    if not path.exists():
        raise ValueError(f"{path} does nto exist")
    radar_dataset = ausLib.read_radar_zipfile(path, first_file=first_file,
                                              concat_dim=concat_dim,
                                              **load_kwargs)
    # hdf5 engine is very slow. but netcdf4 needs dask in single thread mode. Sigh!
    # drop cases where entire field is missing
    if radar_dataset is None:  # no data retrieved.
        my_logger.warning(f'No data for {path}')
        return None
    got_ref = 'reflectivity' in radar_dataset.variables
    if got_ref:
        L = radar_dataset.reflectivity.isnull().all(['x', 'y'])
        if L.sum() > 0:
            bad_times = L[concat_dim][L]
            my_logger.warning(f'All data missing for times: {bad_times.values}')
            radar_dataset = radar_dataset.where(~L, drop=True)
    if len(radar_dataset) == 0:
        my_logger.warning(f'No data for {path}')
        return None
    # sort variables so all dimensions are monotonically increasing.
    result = dict()
    for var in radar_dataset.data_vars:
        v = radar_dataset[var]
        for d in list(v.dims):
            v = v.sortby(d)
        result[var] = v
        my_logger.debug(f'Sorted variable {var}')
    radar_dataset = xarray.Dataset(result)

    # do some specials for reflectivity
    v='reflectivity'
    if got_ref:
        with xarray.set_options(keep_attrs=True):
            units = ausLib.lc_none(radar_dataset[v].attrs, 'units')
            std_name = ausLib.lc_none(radar_dataset[v].attrs, 'standard_name')
            if std_name != 'equivalent_reflectivity_factor':
                ValueError(f'Std name for reflectivity is {std_name} not equivalent_reflectivity_factor')
            # count number of missing values but only for reflectivity data
            miss_var = str(v) + '_fract_missing'
            vars_non_time = set(list(radar_dataset[v].dims)) - {concat_dim}
            mv = radar_dataset[v].isnull()
            mv = mv.sum(vars_non_time) / mv.count(vars_non_time)
            if (mv >1).any():
                breakpoint()
            mv = mv.assign_attrs(units='fraction', extra='fraction of missing values')
            radar_dataset[miss_var] = mv

            # set values below threshold to 0 and above to missing for reflectivity
            L0 = None
            if (dbz_ref_limits is not None) and (units == 'dbz') and (std_name == 'equivalent_reflectivity_factor'):
                L0 = radar_dataset[v] < dbz_ref_limits[0]
                L1 = radar_dataset[v] > dbz_ref_limits[1]
                vars_non_time = set(list(L1.dims)) - {concat_dim}
                miss_var = str(v) + '_fract_high'
                radar_dataset[miss_var] = L1.sum(vars_non_time) / L1.count(
                    vars_non_time). \
                    assign_attrs(units='fraction', threshold_dbz=dbz_ref_limits[1])  # fraction of values > thresh
                radar_dataset[miss_var] = radar_dataset[miss_var]
                if L1.sum() > 0: # got some values above the max thresh
                    radar_dataset[v] = radar_dataset[v].where(~L1, other=np.nan)  # above threshold nan

                    my_logger.debug(f'Set {L1.values.sum():,} values above {dbz_ref_limits[1]} to missing for {v}')
            if units == 'dbz':
                radar_dataset[v] = 10 ** (radar_dataset[v] / 10.) # convert from DBZ to Z
                if L0 is not None:  # have a mask for 0
                    radar_dataset[v] = radar_dataset[v].where(~L0, other=0.0).assign_attrs(zero_dbz=dbz_ref_limits[0])  # below threshold 0
                    my_logger.debug(f'Set {L0.values.sum():,} values below  {dbz_ref_limits[0]} to 0 ')
                radar_dataset[v].attrs['units'] = 'mm**6/m**3'

    if coarsen is not None:  # coarsen the data
        if coarsen_cv_max is not None:
            speckle_vars=tuple('reflectivity')
        else:
            speckle_vars=()
        radar_dataset = ausLib.coarsen_ds(radar_dataset, coarsen, bounds_vars=bounds_vars,
                                          coarsen_method=coarsen_method, check_finite=check_finite,
                                          speckle_vars=speckle_vars)

        for bname in speckle_vars: # loop over speckle vars -- vars we generated speckle for.
            vname = 'reflectivity_speckle'
            miss_var = f'fract_{vname}'
            cv = radar_dataset[vname] / radar_dataset[bname]
            msk = cv > coarsen_cv_max
            if msk.sum() > 0:
                radar_dataset[bname] = radar_dataset[bname].where(~msk, other=np.nan)
                my_logger.warning(f'Set CV {msk.values.sum():,} values above {coarsen_cv_max} to missing for {bname}')
            vars_non_time = set(list(msk.dims)) - {concat_dim}

            v = msk.sum(vars_non_time) / msk.count(vars_non_time)  # fraction of values above threshold
            radar_dataset[miss_var] = v.assign_attrs(units='fraction',extra=f'fraction coarsened CV > {coarsen_cv_max}')

    radar_dataset = radar_dataset.compute()
    return radar_dataset


def read_multi_zip_files(zip_files: typing.List[pathlib.Path],
                         coarsen: typing.Optional[typing.Dict[str, int]] = None,
                         coarsen_method: type_cm = 'mean',
                         dbz_ref_limits: typing.Optional[typing.Tuple[float, float]] = None,
                         coarsen_cv_max: typing.Optional[float] = None) -> xarray.Dataset:
    # read in first file to get helpful co-ord info. The more we read the slower we go.
    drop_vars_first = ['error', 'reflectivity']  # variables not to read for meta info
    drop_vars = ['error', 'x_bounds', 'y_bounds', 'proj']  # variables not to read for data
    fld_info = read_zip(zip_files[0], drop_variables=drop_vars_first, coarsen=coarsen,
                        first_file=True, parallel=True)

    ds = []
    for zip_file in zip_files:
        dd = read_zip(zip_file, coarsen=coarsen, coarsen_method=coarsen_method,
                      dbz_ref_limits=dbz_ref_limits,
                      coarsen_cv_max=coarsen_cv_max,
                      drop_variables=drop_vars, concat_dim='valid_time', parallel=True,
                      combine='nested',
                      chunks=dict(valid_time=6), engine='netcdf4')
        ds.append(dd)
    my_logger.info(f'Read in {len(ds)} files {ausLib.memory_use()}')
    ds = xarray.concat(ds, dim='valid_time').rename(valid_time='time').sortby('time')

    my_logger.debug(f'Concatenated data   {ausLib.memory_use()}')
    # merge seems to generate huge memory footprint so just copy across the data from fld_info
    ds = ds.drop_vars('n2', errors='ignore')
    fld_info = fld_info.drop_vars('n2', errors='ignore')
    for v in fld_info.data_vars:
        if v not in ds.data_vars:
            ds[v] = fld_info[v]
            my_logger.debug(f'Added variable {v} to ds')
        else:
            my_logger.debug(f'Variable {v} already in ds')
    ds = fix_spatial_units(ds).compute()
    return ds


##
if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!

    parser = argparse.ArgumentParser(description='Process reflectivity data')
    parser.add_argument('site', help='Radar site to process',
                        default='Melbourne', choices=site_numbers.keys())
    parser.add_argument('--year', nargs=2, type=int, help='range of years to process', default=[2020, 2023])
    parser.add_argument('--months', nargs='+', type=int, help='list of months to process', default=range(1, 13))
    parser.add_argument('-v', '--verbose', action='count', help='Verbose output', default=0)
    parser.add_argument('outdir', help='output directory. Must be provided')
    parser.add_argument('--glob', help='Pattern for globbing zip files',
                        default='[0-9][0-9].gndrefl.zip')
    parser.add_argument('--resample', nargs='+',
                        help='resample periods for data', default=['30min', '1h', '2h'])
    parser.add_argument('--no_over_write', action='store_true', help='Do not overwrite existing files')
    parser.add_argument('--dask', action='store_true', help='Start dask client')
    parser.add_argument('--log_file',
                        help='Name of log file -- if provided. log info gots there as well as std out/err')
    parser.add_argument('--coarsen', nargs=2, type=int, help='coarsen values for x and y in that order')
    parser.add_argument('--coarsen_method', help='method to use for coarsening', default='mean',
                        choices=['mean', 'median'])
    parser.add_argument('--dbz_range', nargs=2, type=float, default=[15., 55.],
                        help='range for dbz ref. Values below are set to 0 when converting to ref; above to missing')
    parser.add_argument('--cv_max', type=float, help='Maximum coefficient of variability for coarsened data',default=10.)

    args = parser.parse_args()
    # deal with verbosity!
    if args.verbose > 1:
        level = 'DEBUG'
    elif args.verbose > 0:
        level = 'INFO'
    else:
        level = 'WARNING'
    ausLib.init_log(my_logger, level=level, log_file=args.log_file, mode='w')
    # print out all the arguments
    for name, value in vars(args).items():
        my_logger.info(f"Arg:{name} =  {value}")
    if args.dask:
        my_logger.info('Starting dask client')
        client = ausLib.dask_client()
    else:
        dask.config.set(scheduler="single-threaded")  # make sure dask is single threaded.
        my_logger.info('Running single threaded')
    my_logger.info(f'Processing reflectivity data for {args.site} for {args.year[0]} to {args.year[1]} {memory_use()}')
    my_logger.info(f'Output dir is {args.outdir}')
    site_number = f'{site_numbers[args.site]:d}'
    indir = pathlib.Path('/g/data/rq0/hist_gndrefl') / site_number
    my_logger.info(f'Input directory is {indir}')
    my_logger.info(f'resample periods are: {args.resample}')
    if not indir.exists():
        my_logger.warning('Input directory {indir} does not exist')
        raise FileNotFoundError(f'Input directory {indir} does not exist')
    outdir = pathlib.Path(args.outdir) / args.site
    outdir.mkdir(parents=True, exist_ok=True)
    drop_vars_first = ['error', 'reflectivity']  # variables not to read for meta info
    drop_vars = ['error', 'x_bounds', 'y_bounds', 'proj']  # variables not to read for data
    if args.coarsen:
        coarsen = dict(x=args.coarsen[0], y=args.coarsen[1])
    else:
        coarsen = None
    for year in range(*args.year):
        my_logger.info(f'Processing year {year}')
        for month in args.months:
            pattern = f'{site_number}_{year:04d}{month:02d}' + args.glob
            data_dir = indir / f'{year:04d}'
            zip_files = sorted(data_dir.glob(pattern))
            if len(zip_files) == 0:
                my_logger.info(f'No files found for  pattern {pattern} in {data_dir} {ausLib.memory_use()}')
                continue
            my_logger.info(f'Found {len(zip_files)} files for pattern {pattern} {ausLib.memory_use()} ')
            file = f'hist_gndrefl_{year:04d}_{month:02d}.nc'
            outpath = outdir / file
            if args.no_over_write and outpath.exists():
                my_logger.warning(f'{outpath} and no_over_write set. Skipping processing')
                continue

            ds = read_multi_zip_files(zip_files, dbz_ref_limits=tuple(args.dbz_range),
                                      coarsen=dict(x=4, y=4), coarsen_method='mean', coarsen_cv_max=args.cv_max)
            my_logger.info(f'Loaded data for {year}-{month} {ausLib.memory_use()}')  # sadly can  cause OOM fail...
            attr_var = ds.drop_vars(['reflectivity', 'reflectivity_speckle', 'error'], errors='ignore').\
                mean('time', keep_attrs=True)
            ref_summ = summary_process(ds.reflectivity, mean_resample=args.resample, threshold=0.5, base_name='reflectivity')
            attr_var = attr_var.expand_dims(time=ref_summ.time)
            ref_summ = ref_summ.merge(attr_var) # merge in attributes data.
            min_res = ref_summ.sample_resolution / np.timedelta64(1, 'm')
            # check sample resolution is reasonable
            if min_res < 5:
                ValueError(f'small sample resolution {min_res} mins')
            elif min_res > 15:
                ValueError(f'large sample resolution {min_res} mins')

            ref_summ.encoding.update(zlib=True, complevel=4)
            ref_summ.time.assign_attrs(units='minutes since 1990-01-01T00:00')
            my_logger.info(f"computed month of summary data {memory_use()}")
            my_logger.info(f'Writing summary data to {outpath} {ausLib.memory_use()}')
            ref_summ.to_netcdf(outpath, unlimited_dims='time')
