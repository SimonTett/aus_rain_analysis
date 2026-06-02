#!/usr/bin/env python
# process reflectivity data. Iterates over data processing a month at a time worth of data.
import pathlib
import typing
import argparse
import numpy as np
import xarray
import ausLib
from ausLib import memory_use, my_logger, site_numbers
import pandas as pd
import multiprocessing




##
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
                    mean_resample: typing.Union[typing.List[str], str,None] = None,
                    time_dim: str = 'time',
                    threshold: typing.Optional[float] = None,
                    base_name: typing.Optional[str] = None,
                    min_fract_avg: float = 1.0,
                    max_sample_resolution: typing.Optional[pd.Timedelta] = None,
                    subsample: typing.Optional[pd.Timedelta] = None) -> typing.Optional[xarray.Dataset]:
    f"""
    Process data_array for max (and mean & time of max).

    Args:
        data_array: dataArray to be processed. 
        mean_resample: the resample period(s) to generate means.
        base_name: basename to be used. If not provided then long_name in attrs of data_array will be used. 
        time_dim: the name of the time dimension
        threshold: the threshold above which var is used in *_thresh dataArrays. 
        min_fract_avg: minimum fraction of data that must be available for a time period. 
          If less than this then the time period is dropped.
        If not provided All _thresh vars will be empty
        subsample: if set then resample the data to this period. If None then no resampling is done.
        max_sample_resolution: maximum sample resolution. 
          If set and the sample resolution is greater than max_sample_resolution 
          then sample_resolution will be set to this value.


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
            :param subsample: 


    """

    time_bounds = [data_array[time_dim].min().values, data_array[time_dim].max().values]
    # if subsampling gen times
    if subsample:  # want to sub-sample the data.
        my_logger.info(f'Resampling to {subsample} time_bounds:{time_bounds} len: {len(data_array[time_dim])}')
        times = pd.date_range(start=time_bounds[0], end=time_bounds[1],
                              freq=subsample, inclusive='both')  # times wanted
        data_array = data_array.reindex({time_dim: times}, method='nearest', tolerance=subsample / 2)
    # remove slices where ALL data is missing
    L = data_array.isnull().all(['x', 'y'])  # All values missing mask in time.
    data_array = data_array.where(~L, drop=True)
    # and update the time_bounds
    time_bounds = [data_array[time_dim].min().values, data_array[time_dim].max().values]

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
                   f'mean_{base_name}', f'count_raw_{base_name}', f'count_raw_miss_{base_name}'
                   ]
    if threshold is not None:  # add on threshold vars
        vars_to_gen += [f"median_{base_name}_thresh", f"count_raw_{base_name}_thresh",
                        f"mean_{base_name}_thresh", f"count_{base_name}_thresh"]
    # list of variables that won't have resample_prd included.
    no_resample_vars = [var for var in vars_to_gen if 'raw' in var]
    no_resample_vars += ['time_bounds', 'sample_resolution']

    result = empty_ds(data_array.isel({time_dim: 0}, drop=True), mean_resample,
                      vars_to_gen, f"time_max_{base_name}",
                      no_resample_vars=no_resample_vars)
    # add in counts of samples
    dummy = xarray.DataArray(data=np.repeat(np.nan, len(mean_resample)),
                             coords=dict(resample_prd=mean_resample))
    for n in ['count', 'expected_samples']:  # any other vars that are time only for resample prds.
        result[f'{n}_{base_name}'] = dummy
    # and time_bounds
    result['time_bounds'] = time_bounds
    count_raw = data_array[time_dim].count()
    result[f'count_raw_{base_name}'] = count_raw
    result[f'count_raw_miss_{base_name}'] = data_array.isnull().sum(time_dim)
    # add in time resolution and complain (but continue) if have values diff from median
    td = data_array.time.diff(time_dim)
    time_resoln = td.median().values
    if (max_sample_resolution is not None) and (time_resoln > max_sample_resolution):
        my_logger.warning(
            f"Sample resolution {time_resoln} > {max_sample_resolution}. Setting to {max_sample_resolution}")
        time_resoln = max_sample_resolution
    result['sample_resolution'] = time_resoln

    L = (td != time_resoln)
    if L.any():
        scale = np.timedelta64(1, 'm')
        res = time_resoln / scale
        bad_td = td.where(L, drop=True) / scale
        my_logger.warning(
            f"Time resolution not consistent -- median {res} mins. {len(bad_td)} bad values {np.unique(bad_td.values)}")
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
        resample_dict = {time_dim: mn_resamp, 'closed': 'right', 'label': 'right'}
        # for consistency with sub-daily precip -- time-samples are at the *end* of the record.
        mean_resamp = (data_array.resample(**resample_dict).mean(skipna=True).
                       expand_dims(resample_prd=[mn_resamp]))
        count_resamp = (data_array[time_dim].resample(**resample_dict).count().
                        expand_dims(resample_prd=[mn_resamp]))
        # work out the expected number of samples. If have less than this then mask the data.
        # Can have more as sample_resolution is the median estimator of the diff.
        expected_samps = np.floor(pd.to_timedelta(mn_resamp).total_seconds() / result['sample_resolution'].dt.seconds)
        expected_samps = expected_samps.assign_attrs(extra='Expected number of samples')
        ok = (count_resamp >= min_fract_avg * expected_samps.values)
        count_enough_samples = ok.sum(time_dim, keep_attrs=True). \
            assign_attrs(long_name='Count_samples',
                         extra='Number of samples with sufficient data')
        if count_enough_samples < 1:  # not enough data to compute a single time mean
            my_logger.warning(f'Not enough data to compute any time means for '
                              f'resample {mn_resamp} and times {time_str}')
            continue  # will get default values for this case.
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

        rr.update({
            f"max_{base_name}": max_mean_resamp,
            f"median_{base_name}": median_mean_resamp,
            f"mean_{base_name}": mean_mean_resamp,
            f"time_max_{base_name}": time_max_resamp,
            f'count_{base_name}': count_enough_samples,
            f"expected_samples_{base_name}": expected_samps
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
    if len(list_resamp_result) > 0:  # some  data to process

        rr = xarray.concat(list_resamp_result, dim='resample_prd')
        rr = rr.assign_coords(resample_prd=[str(s) for s in rr['resample_prd'].values]) # fix the coord values.
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
        elif ('sample' in k) or (k in ['time_bounds']):  # vars to leave alone!
            pass
        else:
            comp = str(k).split('_')[0]
            result[k].attrs['long_name'] = comp + " " + base_name

    result['threshold'] = threshold
    result['max_fraction'] = min_fract_avg
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
def read_multi_zip_files(zip_files: typing.List[pathlib.Path],
                         coarsen: typing.Optional[typing.Dict[str, int]] = None,
                         coarsen_method: ausLib.type_cm = 'mean',
                         dbz_ref_limits: typing.Optional[typing.Tuple[float, float]] = None,
                         #coarsen_cv_max: typing.Optional[float] = None,
                         region: typing.Optional[typing.Dict[str, slice]] = None,
                         to_rain: ausLib.type_rain_conv = None,
                         calibration: typing.Optional[pd.DataFrame] = None,
                         ) -> xarray.Dataset:
    """

    :param zip_files:
    :param coarsen:
    :param coarsen_method:
    :param dbz_ref_limits:
    :param coarsen_cv_max:
    :param region:
    :param to_rain:
    :param calibration:
    :return:
    """
    # read in first file to get helpful co-ord info. The more we read the slower we go.
    drop_vars_first = ['error', 'reflectivity', 'doppler_velocity',
                       'count_rain_rate_missing']  # variables not to read for meta info
    drop_vars = ['error', 'x_bounds', 'y_bounds', 'proj', 'doppler_velocity',
                 'count_rain_rate_missing']  # variables not to read for data
    fld_info = ausLib.read_zip(zip_files[0], drop_variables=drop_vars_first, coarsen=coarsen,
                        first_file=True, parallel=True, region=region)

    ds = []
    for zip_file in zip_files:
        dd = ausLib.read_zip(zip_file, coarsen=coarsen, coarsen_method=coarsen_method,
                      dbz_ref_limits=dbz_ref_limits,
                      region=region,
                      drop_variables=drop_vars, concat_dim='valid_time',
                      combine='nested',
                      to_rain=to_rain,
                      calibration=calibration)

        ds.append(dd)
    my_logger.info(f'Read in {len(ds)} files {ausLib.memory_use()}')
    ds = xarray.concat(ds, dim='valid_time').rename(valid_time='time').sortby('time')

    my_logger.debug(f'Concatenated data   {ausLib.memory_use()}')
    # merge seems to generate huge memory footprint so just copy across the data from fld_info
    ds = ds.drop_vars('n2', errors='ignore')
    fld_info = fld_info.drop_vars('n2', errors='ignore').rename(valid_time='time').sortby('time')
    for v in fld_info.data_vars:
        if v not in ds.data_vars:
            ds[v] = fld_info[v]
            my_logger.debug(f'Added variable {v} to ds')
        else:
            my_logger.debug(f'Variable {v} already in ds')
    ds = fix_spatial_units(ds).compute()
    # add in the long/lat coords
    ds = ausLib.add_long_lat_coords(ds)
    return ds

def mask_anaprop(radar_ds:xarray.Dataset,anaprop:xarray.DataArray) -> xarray.Dataset:
    """
    Mask radar_ds where anaprop is False. Returns a new dataset with the same variables as radar_ds but with timestamps removed where anaprop
    :param radar_ds: dataset to mask by anaprop. Must have a time dimension
    :param anaprop: dataarray containing potential anaprop (True= potential anaprop; False = not potential anaprop)

    anaprop is interpolated by nearest neighbour time values to the times of the radar_ds
    """
    times =radar_ds.time.values
    anaprop = anaprop.sel(time=times,method='nearest').assign_coords(time=times) # reindex anaprop to radar_ds times. Use nearest neighbour interpolation.
    anaprop.load()
    result = radar_ds.where(~anaprop, drop=True) # remove anaprop affected times
    my_logger.info(f'Anaprop filter: # orig times: {len(radar_ds.time)}, now: {len(result.time)}')
    return result



##
if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!

    parser = argparse.ArgumentParser(description='Process reflectivity data')
    ## the next three args could be standard and processed by the std processing.
    parser.add_argument('site', help='Radar site to process',
                        default='Melbourne', choices=site_numbers.keys())
    parser.add_argument('outdir', type=pathlib.Path,
                        help='output directory. Will be created if it does not exist. If not set will be cwd()/site ',
                        nargs='?')
    parser.add_argument('--indir', type=pathlib.Path, help='input directory',
                        default = ausLib.hist_ref_dir)


    parser.add_argument('--years', nargs='+', type=int, help='List of years to process',
                        default=range(2020, 2023))
    parser.add_argument('--months', nargs='+', type=int, help='list of months to process', default=range(1, 13))

    parser.add_argument('--glob', help='Pattern for globbing zip files',
                        default='[0-9][0-9].gndrefl.zip')
    parser.add_argument('--resample', nargs='+',
                        help='resample periods for data', default=['30min', '1h', '2h'])
    parser.add_argument('--coarsen', nargs=2, type=int, help='coarsen values for x and y in that order')
    parser.add_argument('--coarsen_method', help='method to use for coarsening', default='mean',
                        choices=['mean', 'median'])
    parser.add_argument('--dbz_range', nargs=2, type=float, default=[15., 55.],
                        help='range for dbz ref. '
                             'Values below are set to 0 when converting DBZ to linear units; above to missing')
    #parser.add_argument('--cv_max', type=float, help='Maximum coefficient of variability for coarsened data',
    #                    default=None)
    parser.add_argument('--write_full', action='store_true',
                        help='Write out full datasets after coarsening. Will be written out to site_full/filename')
    parser.add_argument('--region', nargs=4, type=float, help='Region to extract data for as x0 x1 y0 y1')
    parser.add_argument('--min_fract_avg', type=float,
                        help='Minimum fraction of data present when generating averages',
                        default=1.0)
    parser.add_argument('--threshold', type=float,
                        help='Threshold for reflectivity. Used in threshold vars', default=0.5)
    parser.add_argument('--extract_coords_csv',
                        help="""CSV file with coordinates (Longitude & Latitude) to extract data for.
                        CSV file should be readable with ausLib.read_gsdr_csv.
                        Data will be put in outdir/site_coord/filename""")
    parser.add_argument('--subsample', type=pd.Timedelta,
                        help='Timedelta to sub-sample to -- should be parsed by pd.Timedelta')
    parser.add_argument('--to_rain', type=float, nargs=2, help='Convert Reflectivity to rain using R=c[0]Z^c[1]')
    parser.add_argument('--max_sample_resolution', type=pd.Timedelta, help='Maximum sample resolution.')
    parser.add_argument('--use_rainfields3', action='store_true',
                        help='Use rainfields3 data rather than hist. Will need calibration values '
                             'from meta-data')
    parser.add_argument('--quantize_levels',type=float,nargs='+',
                        help='Levels to quantize too.')
    parser.add_argument('--anaprop',type=pathlib.Path,
                        help='Netcdf file containing anaprop data. anaprop variable should be anaprop. Will also modify outdir to include _anaprop if outdir not defined')
    ausLib.add_std_arguments(parser)
    args = parser.parse_args()
    my_logger = ausLib.process_std_arguments(args)  # deal with the std arguments

    time_unit = 'minutes since 1970-01-01'  # units for time in output files

    # print out all the arguments and add them to attributes of the final dataset.

    extra_attrs = dict(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.now('UTC').isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()],
                       site=args.site, dbz_range=args.dbz_range, min_fract_avg=args.min_fract_avg)
    if args.to_rain is not None:
        extra_attrs.update(to_rain=args.to_rain)
    if args.coarsen is not None:
        extra_attrs.update(coarsen=args.coarsen, coarsen_method=args.coarsen_method)

    site_number = f'{site_numbers[args.site]:d}'
    indir = args.indir/ site_number
    if args.use_rainfields3:
        indir = pathlib.Path('/g/data/rq0/rainfields3') / site_number
    my_logger.info(f'Input directory is {indir}')

    if not indir.exists():
        my_logger.warning('Input directory {indir} does not exist')
        raise FileNotFoundError(f'Input directory {indir} does not exist')
    my_logger.info(f'resample periods are: {args.resample}')

    ## deal with some args.  This could (eventually) go into process_std_args
    indir = args.indir # if use rainfall3 then this varies.
    glob = args.glob # if use rainfall3 then this varies.
    site_number = f'{ausLib.site_numbers[args.site]:d}'

    indir = indir/site_number # work out directory for input data. Depends on site_number.
    my_logger.info(f'Input directory is {indir}')
    if not indir.exists():
        my_logger.warning('Input directory {indir} does not exist')
        raise FileNotFoundError(f'Input directory {indir} does not exist')

    outdir = args.outdir
    if outdir is None:
        outdir = pathlib.Path.cwd()/args.site
        if args.anaprop:
            outdir = outdir.parent/(outdir.name+'_anaprop') # add _anaprop onto dir name

    outdir.mkdir(parents=True, exist_ok=True)
    my_logger.info(f'Output directory is {outdir}')


    outdir_full = None
    out_coord_dir = None
    if args.write_full:
        outdir_full = outdir.parent / (outdir.name + '_full')
        outdir_full.mkdir(parents=True, exist_ok=True)
        my_logger.info(f'Full data will be written to {outdir_full}')
    coord_df = None
    if args.extract_coords_csv:  # data going out
        coord_df = ausLib.read_gsdr_csv(args.extract_coords_csv)
        out_coord_dir = outdir.parent / (outdir.name + '_coord')
        out_coord_dir.mkdir(parents=True, exist_ok=True)
        my_logger.info(f'Extracted coord data will be written to {out_coord_dir}')
    extra_attrs.update(outdir=str(outdir), outdir_full=str(outdir_full), out_coord_dir=str(out_coord_dir),
                       site=args.site)
    drop_vars_first = ['error', 'reflectivity']  # variables not to read for meta info
    drop_vars = ['error', 'x_bounds', 'y_bounds', 'proj']  # variables not to read for data
    if args.coarsen:
        coarsen = dict(x=args.coarsen[0], y=args.coarsen[1])
    else:
        coarsen = None
    if args.region:
        region = dict(x=slice(*args.region[0:2]), y=slice(*args.region[2:]))
        my_logger.info(f'Region is {region}')
    else:
        region = None
    calib = None
    if args.use_rainfields3:
        my_logger.info('Loading rainfields3 bias adjustment')
        raise NotImplementedError('Need to add in calibration values for rainfields3')
    if args.anaprop:
        ds_anaprop = xarray.open_dataset(args.anaprop)
        if ds_anaprop.attrs['site'] != args.site:
            raise ValueError(f"Site: {ds_anaprop.attrs['site']} in anaprop file {args.anaprop} does not match site argument {args.site}")
        anaprop =ds_anaprop['anaprop'] # Loading happens for the chunk we need when filtering is done.
        # check site
        my_logger.info(f'Loaded anaprop data from {args.anaprop} with {len(anaprop.time)} time steps')
    for year in args.years:
        my_logger.info(f'Processing year {year}')
        for month in args.months:
            pattern = f'{site_number}_{year:04d}{month:02d}' + args.glob
            if args.use_rainfields3:
                file = f'rainfields3_gndrefl_{year:04d}_{month:02d}'
                data_dir = indir / f'{year:04d}' / 'gndrefl'
                raise NotImplementedError('Need to add in calibration values for rainfields3')
            else:
                file = f'hist_gndrefl_{year:04d}_{month:02d}'
                data_dir = indir / f'{year:04d}'

            zip_files = sorted([pathlib.Path(p) for p in data_dir.glob(pattern)])
            if len(zip_files) == 0:
                my_logger.info(f'No files found for   {data_dir}/{pattern} in {data_dir} {ausLib.memory_use()}')
                continue
            my_logger.info(f'Found {len(zip_files)} files for {data_dir}{pattern} {ausLib.memory_use()} ')
            if args.to_rain:
                file += '_rain'
            file += '.nc'
            outpath = outdir / file
            if (not args.overwrite) and outpath.exists():
                my_logger.warning(f'{outpath} exists  skipping processing. Use --overwrite')
                continue
            to_rain: ausLib.type_rain_conv = None
            if args.to_rain is not None:
                to_rain = tuple(args.to_rain)  # type checker moaning here.

            ds = read_multi_zip_files(zip_files,
                                      dbz_ref_limits=(args.dbz_range[0], args.dbz_range[1]),
                                  coarsen=coarsen, region=region,
                                  coarsen_method=args.coarsen_method,
                                  to_rain=to_rain,
                                  calibration=calib)
            my_logger.info(f'Loaded data for {year}-{month} {ausLib.memory_use()}')

            # handle anaprop
            if args.anaprop:
                orig_times = len(ds.time)
                ds = mask_anaprop(ds,anaprop=anaprop)
                my_logger.info(f'Filtered out anaprop data for {year}-{month} {ausLib.memory_use()}')

            # compute summaries
            basename = 'reflectivity'
            if args.to_rain is not None:
                basename = 'rain_rate'
            summary_data = summary_process(ds[basename], mean_resample=args.resample,
                                           threshold=args.threshold,
                                           base_name=basename,
                                           min_fract_avg=args.min_fract_avg,
                                           subsample=args.subsample,
                                           max_sample_resolution=args.max_sample_resolution)
            my_logger.info(f"computed month of summary data {memory_use()}")
            attr_var = ds.drop_vars([basename, f'{basename}_speckle', 'error'],
                                    errors='ignore').mean('time', keep_attrs=True)
            attr_var = attr_var.expand_dims(time=summary_data.time)
            summary_data = summary_data.merge(attr_var)  # merge in attributes data.
            min_res = summary_data.sample_resolution.dt.seconds / 60.
            # check sample resolution is reasonable
            if min_res < 4:
                ValueError(f'small sample resolution {min_res} mins')
            elif min_res > 15:
                ValueError(f'large sample resolution {min_res} mins')
            ausLib.write_out(summary_data, outpath, time_unit=time_unit, extra_attrs=extra_attrs)
            my_logger.info(f'Writing summary data to {outpath} {ausLib.memory_use()}')


            if args.write_full:  # write out the full file.
                full_file = outdir_full / file
                ausLib.write_out(ds, full_file, time_unit=time_unit, extra_attrs=extra_attrs)
                my_logger.info(f'wrote full data to {full_file} {ausLib.memory_use()}')

            if args.extract_coords_csv:  # write out co-ords
                coord_file = out_coord_dir / file
                radar_proj = ausLib.radar_projection(ds.proj.attrs)
                coord_da = ausLib.data_array_station_match(ds[basename], radar_proj, coord_df)
                att = attr_var.drop_vars(['longitude', 'latitude', 'y_bounds', 'x_bounds']).squeeze('time', drop=True)
                masked_ds = att.merge(coord_da).drop_dims(['x', 'y'])  # drop the time dimension and merge in the coord
                ausLib.write_out(masked_ds, coord_file, time_unit=time_unit, extra_attrs=extra_attrs)
                my_logger.info(f'wrote coord data to {coord_file} {ausLib.memory_use()}')
