# process reflectivity data
import pathlib
import typing

import numpy as np
import xarray

import ausLib

my_logger = ausLib.my_logger
memory_use = ausLib.memory_use


def summary_process(data_array: xarray.DataArray,
                    mean_resample: typing.Union[typing.List[str], str] = None,
                    time_dim: str = 'time',
                    threshold: typing.Optional[float] = None,
                    base_name: typing.Optional[str] = None,
                    dbz: bool = False) -> typing.Optional[xarray.Dataset]:
    f"""
    Process data_array for max (and mean & time of max).

    Args:
        data_array: dataArray to be processed. 
        dbz: data is reflectivity data in dbz. 
            When means are generated then values will be raised to the power 10, 
            mean computed and then log 10 taken
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
            f'count_{base_name}': count of resampled samples  
            f'count_max_{base_name}': max number of samples in resample. 


    """

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

    time_bounds = [ref.time.min().values, ref.time.max().values]
    time_bounds = xarray.DataArray(time_bounds, dims='bounds').rename('time_bounds')
    time_str = f"{time_bounds.values[0]} - {time_bounds.values[1]}"
    if mean_resample is None:
        mean_resample = ['1h']
    if isinstance(mean_resample, str):
        mean_resample = [mean_resample]
    if base_name is None:
        base_name = data_array.attrs.get('long_name')
        if base_name is None:  # raise an error
            raise ValueError('base_name not provided and not in da.attrs["long_name"]')

    # set up empty result
    vars_to_gen = [f"max_{base_name}", f"median_{base_name}", f"mean_raw_{base_name}",
                   f'mean_{base_name}']
    if threshold is not None:  # add on threshold vars
        vars_to_gen += [f"median_{base_name}_thresh", f"count_raw_{base_name}_thresh",
                        f"mean_{base_name}_thresh", f"count_{base_name}_thresh"]
    # lit of variables that won't have resample_prd included.
    no_resample_vars = [var for var in vars_to_gen if 'raw' in var]
    no_resample_vars += ['time_bounds']

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
    # check we have some data and if not exit!
    bad = data_array.isnull().all(time_dim, keep_attrs=True)  # find out where *all* data null
    if bad.all():
        my_logger.error(f"All missing at {bad[time_dim].isel[[0, -1]]}")
        return result

    my_logger.debug(f"Processing data for {time_str} {memory_use()}")
    # get on with processing data now!
    if dbz:
        data_array = np.exp(data_array)
        if threshold:
            threshold = np.exp(threshold)  # if in dbz have converted everything to 10** and at end will invert this for all vars

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
                        expand_dims(resample_prd=[mn_resamp]))
        max_samps = count_resamp.max()
        ok = (count_resamp == max_samps)
        mean_resamp = mean_resamp.where(ok, drop=True)  # mask where have complete data
        # for max want to have consistent no of values.
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
        list_resamp_result.append(rr)
    # end of looping over time resample periods
    rr = xarray.concat(list_resamp_result, dim='resample_prd')
    result.update(rr)
    # if we are dbz take log 10 of variables!
    if dbz:
        if threshold is not None:
            threshold = np.log(threshold)  # convert threshold back to initial vars.
        # work out variables we will want to take log10 off.
        vars_to_log = set()
        for variable in result.data_vars:
            want_var = base_name in variable
            for start_str in ['count', 'time']:
                want_var = want_var and not variable.startswith(start_str)
            if want_var:
                vars_to_log.add(variable)
        # now to take logs
        for variable in vars_to_log:
            my_logger.debug(f'Taking log of {variable}')
            result[variable] = np.log(result[variable])
    # sort out meta-data.
    variables = [v for v in result.variables]
    for k in variables:
        if k.startswith("time_"):
            comp = k.split('_')[1]
            result[k].attrs['long_name'] = 'time of ' + comp + " " + base_name
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
    result.attrs.update(ds.attrs)  # preserve initial metadata.
    my_logger.info(f"Summaries computed for  {time_str} {memory_use()}")
    return result


# ausLib.dask_client()
outdir = pathlib.Path('/scratch/wq02/st7295/summary_reflectivity')
outdir.mkdir(parents=True, exist_ok=True)
year = 2010
ausLib.init_log(my_logger,level='INFO')
for month in range(1, 13):
    pattern = f'*{year:04d}{month:02d}[0-9][0-9].gndrefl.zip'
    zip_files = sorted(pathlib.Path('/g/data/rq0/hist_gndrefl/4/2010').glob(pattern))
    if len(zip_files) == 0:
        my_logger.info(f'No files found for  pattern {pattern} {ausLib.memory_use()}')
        continue
    my_logger.info(f'Found {len(zip_files)} files for pattern {pattern} {ausLib.memory_use()} ')
    singleton_vars = ['x_bounds', 'y_bounds', 'proj']
    datasets = [ausLib.read_radar_zipfile(zip_file, singleton_vars=singleton_vars) for zip_file in zip_files]
    ds = xarray.concat(datasets, dim='valid_time', data_vars='minimal').rename(valid_time='time')
    my_logger.info(f'Concatenated data. {ausLib.memory_use()}')  # sadly cases OOM fail.
    del datasets  # maybe free up a bit of memory

    my_logger.info(f'Loaded data. Now coarsening..{ausLib.memory_use()}')  # sadly cases OOM fail...
    ref = np.log(np.exp(ds.reflectivity).coarsen(x=4, y=4).mean())  # coarsen data to 2 x 2km
    ##now to process data
    my_logger.info(f'Coarsened. Now processing {ausLib.memory_use()}')
    ref_summ = summary_process(ref, dbz=True, base_name='Reflectivity',
                               mean_resample=['30min', '1h', '2h'], threshold=15.0)
    my_logger.info(f"computed month of summary data {memory_use()}")
    s = ref_summ.time[0].dt.strftime('%Y_%m').values
    file = f'hist_gndrefl_4_{s}.nc'
    outpath = outdir / file
    my_logger.info(f'Writing summary data to {outpath} {ausLib.memory_use()}')
    ref_summ.to_netcdf(outpath, unlimited_dims='time')
    my_logger.info(f'Wrote  summary data to {outpath} {ausLib.memory_use()}')




def process(ref):
    my_logger.info(f'Processing a month {ref.time.values[0]} {memory_use()}')
    regrid = np.log10((10 ** ref.astype('float64')).coarsen(x=4, y=4).mean().load())
    my_logger.info('Coarsened')
    ref_summ = summary_process(regrid, dbz=True, base_name='Reflectivity', mean_resample=['30min', '1h'],
                               threshold=15.0)
    my_logger.info(f"computed month of summary data {memory_use()}")
    return ref_summ

# summary = ds.reflectivity.resample(time='MS').apply(process)
