# Library for australian analysis.
import os
import pathlib
import sys

import numpy as np
import xarray
import logging
import pandas as pd
import typing

radar_dir = pathlib.Path("/g/data/rq0/level_2/")

my_logger = logging.getLogger(__name__)  # for logging
# dict to control logging


# stuff for timezones!
from timezonefinder import TimezoneFinder

timezone_finder = TimezoneFinder()  # instance


def gsdp_metadata(files: typing.Iterable[pathlib.Path]) -> pd.DataFrame:
    """
    Read metadata from multiple files into a DataFame. See read_gdsp_metadata for description of columns.
    Args:
        files (): List of files to read in

    Returns: DataFrame of metadata.
    """
    series = []
    for file in files:
        try:
            s = read_gsdp_metadata(file)
            series.append(s)
            my_logger.info(f"Read metadata from {file}")
        except ValueError:  # some problem with data.
            pass
    df = pd.DataFrame(series)
    return df


def read_gsdp_metadata(file: pathlib.Path) -> pd.Series:
    """
      read metadata from hourly precip file created for  gsdp (Global sub-daily precip).
      Format of header data is:
      Station ID:  AU_090180, -> ID
     Country:  Australia,
     Original Station Number:  090180,
     Original Station Name:  AIREYS INLET,
     Path to original data:  B:/INTENSE data/Original data/Australia/AWS 1min_VIC.zip/HD01D_Data_090180_999999998747801.txt,
     Latitude:  -38.4583,
     Longitude:  144.0883,
     Start datetime:  2007072014, -> Start
     End datetime:  2015082009, -> End
     Elevation:  95.0m, -> height & height units
     Number of records:  70868,
     Percent missing data:  7.45,
     Original Timestep:  1min,
     New Timestep:  1hr, -> Timestep
     Original Units:  mm,
     New Units:  mm, -> units
     Time Zone:  CET, -> original_timezone
     Daylight Saving info:  NA,
     No data value:  -999
     Resolution:  0.20,
     Other:

    """
    result = dict()
    with open(file, 'r') as fh:
        hdr_count = 0
        while True:
            key, value = fh.readline().strip().split(":", maxsplit=1)
            # on first line expect key to be 'Station ID'. if not raise an error.
            if (hdr_count == 0) & (key != 'Station ID'):
                raise ValueError(f"File {file} has strange first line {key}:{value}")
            value = value.strip()
            result[key] = value
            hdr_count += 1
            if key == 'Other':  # end fo header
                break  # all done

    # convert same values to loads
    for key in ['Latitude', 'Longitude', 'Resolution', 'Percent missing data', 'No data value']:
        result[key] = float(result[key])
    # integer vales
    for key in ['Number of records']:
        result[key] = int(result[key])
    # convert ht.
    result['height'] = float(result['Elevation'][0:-1])
    result['height units'] = result['Elevation'][-1]
    # rename stuff
    # add in str version of path
    result['file'] = str(file)
    # Store number of header lines -- makes reading in data easier.
    result['header_lines'] = hdr_count
    rename = {'New Units': 'units',
              'Station ID': 'ID',
              'Start datetime': 'Start',
              'End datetime': 'End',
              'Time Zone': 'original_timezone',
              'New Timestep': 'timestep'}
    for old_name, new_name in rename.items():
        result[new_name] = result.pop(old_name)
    # fix timezone...
    if result['original_timezone'].lower() == 'local standard time':
        tz = timezone_finder.timezone_at(lng=result['Longitude'], lat=result['Latitude'])
        my_logger.debug(f"Fixed local std time to {tz} from long/lat coords")
        result["original_timezone"] = tz
        result['Other'] = result['Other'] + " Changed LST"
    if (result['original_timezone'].lower() == 'cet') and (
            result['ID'].startswith("AU_")):  # aus data with timezone CET is probable LST
        tz = timezone_finder.timezone_at(lng=result['Longitude'], lat=result['Latitude'])
        my_logger.debug(f"Fixed CET to {tz} from long/lat coords")
        result["original_timezone"] = tz
        result['Other'] = result['Other'] + " Changed CET"
    # make Start & End datetime
    start = result.pop('Start')
    start = start[0:4] + "-" + start[4:6] + "-" + start[6:8] + "T" + start[8:]
    end = result.pop('End')
    end = end[0:4] + "-" + end[4:6] + "-" + end[6:8] + "T" + end[8:]
    end = pd.to_datetime(end, yearfirst=True, utc=True)  # .tz_localize(result['timezone']).tz_convert('UTC')
    start = pd.to_datetime(start, yearfirst=True, utc=True)  # .tz_localize(result['timezone']).tz_convert('UTC')
    result['End_time'] = end
    result['Start_time'] = start

    result = pd.Series(result).rename(result['ID'])
    return result


def read_gsdp_data(meta_data: pd.Series) -> pd.Series:
    """
    Read data in given a meta data record
    Args:
        meta_data: meta data record as a pandas series.
    Returns: pandas series with actual information. Missing data is set to null. Will have name = ID

    """
    my_logger.debug(f"Reading data for {meta_data.name} from {meta_data.file}")
    freq_transform = {'1hr': '1h'}  # how we transfer timestep info into pandas freq stings.
    # gen index using number of records
    index = pd.date_range(start=meta_data.Start_time, freq=freq_transform[meta_data.timestep],
                          periods=meta_data['Number of records'])
    # check last index is End_time and if not produce a warning.
    if index[-1] != meta_data.End_time:
        my_logger.warning(f"ID: {meta_data.name}: Last index {index[-1]} != End_time {meta_data.End_time}")

    series = pd.read_csv(meta_data.file, skiprows=meta_data.header_lines, header=None).set_index(index)
    series = series.iloc[:, 0]
    series = series.rename(meta_data.name)  # rename
    series = series.astype('float')
    # set all values which are "No data value" to null
    series = series.replace(meta_data['No data value'], np.nan)

    return series


def read_gsdr_csv(file: typing.Union[pathlib.Path, str]) -> pd.DataFrame:
    """
    Read GSDR metadata saved to .csv file
    :param file: file to read.
    :return: dataframe
    """
    df = pd.read_csv(file,
                     index_col=0, parse_dates=['End_time', 'Start_time'])
    return df


def max_process(ds: xarray.Dataset,
                time_dim: str = 'time',
                minf_mean: float = 0.8) -> typing.Optional[xarray.Dataset]:
    """
    Process dataset for max (and mean & time of max).

    Args:
        minf_mean: The minimum fraction data used to compute a time-mean. Data with a fraction below this are ignored.
        time_dim: the name of the time dimension
        ds: dataset to be processed. Should contain the variables:
            rain -- mean rain
            fraction -- fraction of data present in the mean computation.
            longitude & latitude - longitude and latitude co-ords.
        Note this dataset will be loaded.

    Returns:dataset containing maximum values, time of max values and mean values.
    (or None if no non-missing data).

    """

    ok = ds.fraction >= minf_mean
    max_fraction = ok.mean(time_dim)  # fraction of OK hours.
    rain = ds.rain.sel({time_dim: ok})
    bad = rain.isnull().all(time_dim, keep_attrs=True)  # find out where *all* data null
    if bad.all():
        my_logger.warning(f"All missing at {bad[time_dim].isel[[0, -1]]}")
        return None

    max_rain = rain.max(time_dim, keep_attrs=True, skipna=True).rename('max_rain')
    mean_rain = rain.mean(time_dim, keep_attrs=True, skipna=True).rename('mean_rain')
    indx = xarray.where(bad, 0.0, rain).argmax(dim=time_dim, keep_attrs=True).compute()
    max_time = rain[time_dim].compute().isel({time_dim: indx}).where(~bad).rename(
        'time_max_rain')  # max times for this season

    base_name = max_rain.attrs['long_name']

    max_fraction.attrs['long_name'] = "Fraction present " + base_name
    max_fraction.attrs['units'] = '1'

    result = xarray.Dataset(
        dict(max_rain=max_rain, time_max_rain=max_time, mean_rain=mean_rain, max_fraction=max_fraction))

    result.attrs.update(ds.attrs)
    my_logger.info(f"max computed for  {ds[time_dim][0].values} {memory_use()}")
    return result


def gen_mask(example_data_array: xarray.DataArray,
             radar_range: typing.Tuple[float, float] = (4.2e3, 144.8e3)  # values empirically tuned.
             ) -> xarray.Dataset:
    """
    Compute mask -- True where data should be present, False where not.
    :param example_data_array:
    :param radar_range:
    :return:
    """

    rng = np.sqrt(example_data_array.x.astype('float') ** 2 + example_data_array.y.astype('float') ** 2)
    mask = (rng < radar_range[1]) & (rng > radar_range[0])  # cells close to radar are missing as are those far away
    return mask


def summary_process(ds: xarray.Dataset,
                    mean_resample: str = '1h',
                    time_dim: str = 'time',
                    rain_threshold: float = 0.1,
                    minf_mean: float = 0.8) -> typing.Optional[xarray.Dataset]:
    """
    Process dataset for max (and mean & time of max).

    Args:
        ds: dataset to be processed. Must contain rainrate and isfile.
        mean_resample: the resample period to generate means.
        minf_mean: The minimum fraction data used to compute a time-mean. Data with a fraction below this are ignored when computing summaries.
        time_dim: the name of the time dimension
        rain_threshold: the threshold above which rain is counted and used in the median.



    Returns:dataset containing summary values. These are:
         max_rain, time_max_rain, median_rain, mean_rate. median_rain is only for values > rain_threshold.
    Also includes some count values.
                f'count_{mean_resample}': number of mean non-missing values
            f'samples_{mean_resample}': Total number of values
            rain_count: Count of rain values above threshold
            rain_samples: Total number of possible samples.

    """

    def empty_ds(example_da: xarray.DataArray,
                 non_time_variables: typing.Union[typing.List[str], str],
                 vars_time: typing.Union[typing.List[str], str]
                 ) -> xarray.Dataset:
        """
        Return an empty dataset
        :param example_da:
        :param non_time_variables:
        :param vars_time:
        :return: empty dataset.
        """
        # deal with singleton vars
        if isinstance(non_time_variables, str):
            non_time_variables = [non_time_variables]
        if isinstance(vars_time, str):
            vars_time = [vars_time]

        # generates vars
        result = dict()
        for var in non_time_variables:
            result[var] = example_da.where(False).rename(var)
        for var in vars_time:
            result[var] = example_da.where(False).rename(var).astype('<M8[ns]')
            result[var].attrs.pop('units', None)  # remove the units here.
        result = xarray.Dataset(result)
        return result

    rain = ds.rainrate
    # compute # of times with rain. Hopefully, useful to detect radar "clutter".
    rain_samples = ds.isfile.sum(keep_attrs=True).load()
    rain_samples.attrs['long_name'] = 'Total samples'
    rain_count = (rain > rain_threshold).sum(time_dim, skipna=True, min_count=1, keep_attrs=True)
    rain_count.attrs.update(long_name=f'# times with rain >{rain_threshold}')

    fraction = ds.isfile.resample({time_dim: mean_resample}).mean(skipna=True)
    ok = fraction >= minf_mean
    count_of_mean = ok.astype('int').sum(time_dim).load()  # count of mean samples with enough data.
    samples_of_mean = ok.count()  # count of mean samples
    if int(count_of_mean) == 0:  # no data..
        my_logger.warning(f"No data for {fraction[time_dim][[0, -1]].values}")
        result = empty_ds(rain.isel({time_dim: 0}, drop=True),
                          ["max_rain", "mean_rain", "median_rain"],
                          "time_max_rain")

    else:
        msk = (ds.isfile == 1)
        r = rain.fillna(0.0).where(msk)  # replace nans with 0 and apply mask.
        mean_rain = r.resample({time_dim: mean_resample}).mean(skipna=True)
        my_logger.debug(f"mean computed using {mean_resample} for {mean_rain[time_dim][[0, -1]].values} {memory_use()}")
        rain = mean_rain.sel({time_dim: ok}).load()  # select out where data OK and then compute the mean.
        my_logger.debug(f"rain computed for  {rain[time_dim].values[0]} {memory_use()}")
        bad = rain.isnull().all(time_dim, keep_attrs=True)  # find out where *all* data null
        if bad.all():
            my_logger.error(f"All missing at {bad[time_dim].isel[[0, -1]]}")
        # do the actual computation.
        msk = rain > rain_threshold  # mask for rain threshold.
        median_rain = rain.where(msk).median(time_dim, keep_attrs=True, skipna=True).rename('median_rain')
        max_rain = rain.max(time_dim, keep_attrs=True, skipna=True).rename('max_rain')
        mean_rain = rain.mean(time_dim, keep_attrs=True, skipna=True).rename('mean_rain')
        time_max_rain = rain.idxmax(time_dim, keep_attrs=False, skipna=True).rename('time_max_rain')
        result = xarray.Dataset(dict(median_rain=median_rain, max_rain=max_rain,
                                     mean_rain=mean_rain, time_max_rain=time_max_rain))
    # now have result either from empty case or std one,
    # now set the meta data.
    base_name = ds.rainrate.attrs['long_name'] + f" mean {mean_resample}"
    variables = [v for v in result.variables if v.endswith("_rain")]
    for k in variables:
        if k.startswith("time_"):
            comp = "time of " + k.split('_')[1]
        else:
            comp = k.split('_')[0]
        result[k].attrs['long_name'] = comp + " " + base_name
    # append min for median,
    result.median_rain.attrs['long_name'] += f' for rain > {rain_threshold} mm/h'
    count_of_mean.attrs['long_name'] = f"Count of present {mean_resample} samples " + base_name
    samples_of_mean.attrs['long_name'] = f"{mean_resample} samples " + base_name
    counts_ds = xarray.Dataset(
        {
            f'count_{mean_resample}': count_of_mean,
            f'samples_{mean_resample}': samples_of_mean,
            "rain_count": rain_count,
            "rain_samples": rain_samples
        }
    )
    for v in counts_ds.variables:
        counts_ds[v].attrs.pop('units', None)
    result = result.merge(counts_ds)
    result.attrs.update(ds.attrs)  # preserve initial metadata.
    my_logger.info(f"Summaries computed for  {ds[time_dim][[0, -1]].values} {memory_use()}")
    return result


def process_gsdp_record(gsdp_record: pd.Series,
                        resample_max: str = 'QS-DEC'
                        ) -> pd.DataFrame:
    """

    Args:
        gsdp_record: pandas series containing hourly rainfall
        resample_max: time period over which to resample

    Returns:Dataframe containing (as columns) indexed by time:
      #fraction -- fraction of non-missing data
      max -- value of max
      mean -- mean value
      time_max -- time max.

    """
    resamp = gsdp_record.resample(resample_max)
    count_data = gsdp_record.isnull().resample(resample_max).count()
    fraction = (resamp.count() / count_data).rename('fraction')
    max_rain = resamp.max().rename("max_rain")
    time_max = resamp.apply(pd.Series.idxmax).rename("time_max_rain")
    # fix the type for time_max
    time_max = time_max.astype(gsdp_record.index.dtype)
    mean = resamp.mean().rename("mean_rain")  # need to convert to mm/season?
    total = (mean * count_data).rename('total_rain')
    df = pd.DataFrame([max_rain, mean, total, fraction, time_max]).T
    for s in ['max_rain', 'mean_rain', 'total_rain', 'fraction']:
        df[s] = df[s].astype('float')
    df['time_max'] = df['time_max_rain'].astype(gsdp_record.index.dtype)

    return df


def process_radar(data_set: xarray.Dataset, mean_resample: str = '1h', min_mean: float = 0.8,
                  summary_resample: str = 'QS-DEC', time_dim: str = 'time',
                  mask: typing.Optional[xarray.DataArray] = None,
                  radar_range: typing.Tuple[float, float] = (4.2e3, 144.8e3)) -> xarray.Dataset:
    """
    Compute max rainfall on mean rainfall from 6 minute radar rainfall data.

      :param data_set: dataset to be processed. Must contain rainrate, isfile, latitude and longitude
      :param mean_resample: time period to resample data to compute mean
      :param min_mean: min fraction of samples to compute mean.
      :param summary_resample : time period to resample meaned data to compute summary data.
      :param time_dim:dimension that is time
      :param mask: Bool mask. Where False summary data will be set to missing.
      :param radar_range: 2 element tuple, element o is distance from centre to be in radar coverage -- element 1 the max radar coverage.
        missing values inside this range  will be set to zero if isfile is 1 but only if mask not provided.
      :return: summary data.


      """
    # TODO add in time-bounds and change time co-ord to be the centre of the interval.
    # Takes about 2 mins to process a year of 1km radar data to monthly data.
    my_logger.info(f"Starting data processing. {memory_use()}")

    if mask is None:
        mask = gen_mask(data_set.rainrate, radar_range=radar_range)

    resamp = data_set.drop_vars(['longitude', 'latitude']).resample({time_dim: summary_resample})
    result = resamp.map(summary_process, shortcut=True, mean_resample=mean_resample, time_dim=time_dim,
                        minf_mean=min_mean)

    my_logger.debug(f"Processed data and loading data {memory_use()}")
    coords = dict()
    for c in ['latitude', 'longitude']:
        try:  # potentially generating multiple values so just want first one
            coords[c] = data_set[c].isel({time_dim: 0}).squeeze(drop=True).drop_vars(time_dim).load()
        except ValueError:  # no time so jus have the coord
            coords[c] = data_set[c].load()

    variables = [v for v in result.variables if (v != 'x' and 'x' in result[v].coords)]  # things that are spatial
    for var in variables:
        result[var] = result[var].assign_coords(**coords)
        if mask is not None:
            result[var] = result[var].where(mask)
        my_logger.debug(f"added long/lat coords to {var}")
    my_logger.debug(f"Added co-ordinates {memory_use()}")

    my_logger.debug(f"Set attributes {memory_use()}")
    result.attrs.update(data_set.attrs)
    logging.info(f"processed data for {memory_use()}")
    return result


def memory_use() -> str:
    """

    Returns: a string with the memory use in Gbytes or "Mem use unknown" if unable to load resource
     module.

    """
    try:
        import resource
        mem = f"Mem = {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6} Gbytes"
    except ModuleNotFoundError:
        mem = 'Mem use unknown'
    return mem


def resample_max(data_array: xarray.DataArray,
                 resample: str = '1h',
                 time_dim: str = 'time',
                 fillna: typing.Optional[float] = None) -> xarray.Dataset:
    """

    :param data_array: data array to be processed.
    :param resample: resample value 
    :param time_dim: name of time dimension
    :param fillna -- if not Null value to fill nan values with.
    :return: DataSet of processed dataArrays. Computation is resample mean then seasonal max.
     Variables are:
         Max_{dataArray.name}  (max value in a season)
         Max_time_{dataArray.name}  (time of max value in a season)
    """

    result = dict()

    # Can have a bunch of co-ordinates with co-ord time_dim. 
    # Want to remove those as they no longer make sense once we compute the 
    # seasonal max (which reduces the dimension).
    # If wanted they culd be regenerated but xarray has the dt accessor.

    coords_to_remove = [c for c in data_array.coords if
                        (time_dim in data_array[c].coords and data_array[c].name != time_dim)]
    if len(coords_to_remove) > 0:
        s = " ".join(coords_to_remove)
        logging.info(f"Dropping following coords as have {time_dim}: {s}")

    # generate the resample mean values
    rmin = data_array.drop(coords_to_remove)
    if fillna is not None:  # fill nan
        rmin = rmin.fillna(fillna)
    rmin = rmin.resample({time_dim: resample}).mean()
    # now compute the max value for each season.
    seas_resamp = rmin.resample({time_dim: 'QS-DEC'})  # resampler
    # initialise lists.
    mx_time = []
    mx_value = []
    for c, da in seas_resamp:  # loop over resamples/
        bad = da.isnull().all(time_dim, keep_attrs=True)  # find out where *all* data null
        if bad.all():
            my_logger.warning(f"All missing at {c}")
            continue  # skip further processing as all missing.
        # need to load the data as dask does not cope with complex indixing
        da.load()
        my_logger.debug(f"Loaded data for coord {c}. {memory_use()}")
        indx = da.argmax(dim=time_dim, skipna=True, keep_attrs=True)  # index of maxes
        coord = {time_dim: c}  # coordinate to set data time
        mtime = da[time_dim].isel({time_dim: indx}).where(~bad).rename('time_max').assign_coords(
            coord).load()  # max times for this season
        mvalue = da.isel({time_dim: indx}).where(~bad).assign_coords(coord).load()
        # max value for this season
        # then store things.
        mx_time.append(mtime)
        mx_value.append(mvalue)
        my_logger.debug(f"Computed mx value and time of max at time {c}. {memory_use()}")
    mx_time = xarray.concat(mx_time, dim=time_dim).sortby(time_dim)  # concat everything
    mx_value = xarray.concat(mx_value, dim=time_dim).sortby(time_dim)
    my_logger.debug(f"Concatted. {memory_use()}")
    # set up attributes.
    mx_value.attrs = data_array.attrs.copy()
    mx_value.attrs['history'] = mx_value.attrs.get('history', []) + [f"Resampled {resample} + seasonal max"]
    mx_time.attrs = data_array.attrs.copy()
    # remove units (as time coord)
    mx_time.attrs.pop('units', None)
    mx_time.attrs['history'] = mx_time.attrs.get('history', []) + [f"Resampled {resample} + time of seasonal max"]
    key = f"Max_{data_array.name}"
    key_time = f"Max_time_{data_array.name}"
    result[key] = mx_value
    result[key_time] = mx_time
    my_logger.info(f"Stored result size={mx_value.shape} at {key}")

    result = xarray.Dataset(result)  # convert to a dataset.
    return result  # return it.


def summary_radar(files: typing.Union[typing.List[pathlib.Path], pathlib.Path],
                  outfile: typing.Optional[pathlib.Path],
                  mean_resample: str = '1h',
                  summary_resample: str = 'MS',
                  time_dim: str = 'time',
                  chunks: typing.Optional[typing.Dict[str, int]] = None,
                  overwrite: bool = False) -> xarray.Dataset:
    """

    :param files: List of (or single) pathlib.Path's to be read in and processed
    :param outfile: Output path -- where data will be written to. If None nothing will be written to!
    :param mean_resample: resample period for mean
    :param summary_resample: resample period for summary stats (computed on mean_resampled data)
    :param time_dim: Name of time dimension
    :param chunks: Chunk. See xarray.open_dataset for details.
    :param overwrite: If True overwrite existing data.
    :return: dataset of processed data.

    """

    if not overwrite and outfile and outfile.exists():
        raise FileExistsError(f"{outfile} exists and overwrite is False")
    # convert singleton args into lists.

    if isinstance(files, pathlib.Path):
        files = [files]

    if chunks is None:
        chunks = {}

    # open up the files
    input_dataset = xarray.open_mfdataset(files, chunks=chunks, parallel=True)

    # now read them

    ds = process_radar(input_dataset, mean_resample=mean_resample, summary_resample=summary_resample,
                       time_dim=time_dim).compute()

    my_logger.info("Processed  radar")
    encode = dict(zlib=True, complevel=5)
    ds.encoding.update(encode)
    # set the time_max_rain units to days since 1970-0-01
    ds.time_max_rain.encoding['units'] = "hours since 1970-01-01T00:00"

    if outfile:  # want to write data out
        my_logger.info("Writing data out")
        outfile.parent.mkdir(parents=True, exist_ok=True)  # make dir for output
        ds.to_netcdf(outfile, format='NETCDF4')  # write out
        my_logger.info(f"Wrote data to {outfile}")

    return ds  # and return the dataset.


def dask_client() -> 'dask.distributed.Client':
    """
    Start or connect to an existing dask client. Address for server stored in $DASK_SCHEDULER_ADDRESS
    :return: dask.distributed.Client
    """
    import dask.distributed
    try:
        dask_sa = os.environ['DASK_SCHEDULER_ADDRESS']
        my_logger.warning(f"already got client at {dask_sa}")
        client = dask.distributed.get_client(dask_sa, timeout='2s')  # fails. FIX if ever want client
    except KeyError:
        client = dask.distributed.Client(timeout='2s')
        dask_sa = client.scheduler_info()['address']  # need to dig deep into dask doc to get this!
        os.environ['DASK_SCHEDULER_ADDRESS'] = dask_sa
        my_logger.warning(f"Starting new Dask client on {dask_sa}. Available in $DASK_SCHEDULER_ADDRESS ")
    my_logger.warning(f"Dashboard for client at {client.dashboard_link}")
    return client


# set up logging
def init_log(log: logging.Logger,
             level: str,
             log_file: typing.Optional[typing.Union[pathlib.Path, str]] = None,
             mode: str = 'a'):
    """
    Set up logging on a logger! Will clear any existing logging
    :param log: logger to be changed
    :param level: level to be set.
    :param log_file:  if provided pathlib.Path to log to file
    :param mode: mode to open log file with (a  -- append or w -- write)
    :return: nothing -- existing log is modified.
    """
    log.handlers.clear()
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:  %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    # add a file handler.
    if log_file:
        if isinstance(log_file, str):
            log_file = pathlib.Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
        fh = logging.FileHandler(log_file, mode=mode + 't')  #
        fh.setLevel(level)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    log.propagate = False
