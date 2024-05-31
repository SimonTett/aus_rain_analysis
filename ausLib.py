# Library for australian analysis.
import io
import os
import pathlib
import platform
import sys
import tempfile
import time
import zipfile

import requests
# stuff for timezones!
from timezonefinder import TimezoneFinder
import pytz
from datetime import datetime, timedelta
import numpy as np
import xarray
import logging
import pandas as pd
import typing

# dict of site names and numbers.
site_numbers = dict(Adelaide=46, Melbourne=2, Wtakone=52, Sydney=3, Brisbane=50, Canberra=40,
                    Cairns=19, Mornington=36, Grafton=28, Newcastle=4, Gladstone=23
                    )
hostname = platform.node()
if hostname.startswith('gadi'):  # aus super-computer
    radar_dir = pathlib.Path("/g/data/rq0/level_2/")
    data_dir = pathlib.Path("/scratch/wq02/st7295/aus_rain_analysis")
    hist_ref_dir = pathlib.Path("/g/data/rq0/hist_gndrefl/")
elif hostname.startswith('ccrc'):  # CCRC desktop
    data_dir = pathlib.Path("/home/z3542688/data/aus_rain_analysis")
elif hostname == 'geos-w-048':  # my laptop
    data_dir = pathlib.Path(r"C:\Users\stett2\OneDrive - University of Edinburgh\data\aus_radar_analysis")
else:
    raise NotImplementedError(f"Do not know where directories are for this machine:{hostname}")
data_dir.mkdir(exist_ok=True, parents=True)  # make it if need be!

my_logger = logging.getLogger(__name__)  # for logging
# dict to control logging


timezone_finder = TimezoneFinder()  # instance of timezone_finder -- only want one,


def gsdr_metadata(files: typing.Iterable[pathlib.Path]) -> pd.DataFrame:
    """
    Read metadata from multiple files into a DataFame. See read_gdsp_metadata for description of columns.
    Args:
        files (): List of files to read in

    Returns: DataFrame of metadata.
    """
    series = []
    for file in files:
        try:
            s = read_gsdr_metadata(file)
            series.append(s)
            my_logger.info(f"Read metadata from {file}")
        except ValueError:  # some problem with data.
            pass
    df = pd.DataFrame(series)
    return df


def utc_offset(lng: float = 0.0, lat: float = 50.0) -> timedelta:
    """
    Work out the standard time offset from UTC for specified lon/lat co-ord.
    TODO: consider passing in the times so can deal with changing timezones for location.
    Args:
        lng: longitude (in decimal degrees) of point
        lat: latitude (in decimal degrees) of point

    Returns: timedelta from UTC
    Based on example code from chatGPT3.5
    Then various exceptions from tz database are used.
    """
    # uses module var timezone_finder.
    timezone_str = timezone_finder.timezone_at(lng=lng, lat=lat)  # Get the timezone string

    if timezone_str is None:
        raise ValueError(f"Timezone not found for lng:{lng} lat:{lat}")
    # exceptions -- Bureau of Met ignores Eucla (West Aus time) and Broken Hill (NSW time) micro timezones.
    # Also for Giles weather station BoM uses South Aus time
    if timezone_str == 'Australia/Eucla':
        timezone_str = 'Australia/West'  # BoM uses Australian Western Time
        my_logger.info("Setting Eucla tz to Australia/West")
    elif timezone_str == 'Australia/Broken_Hill':
        timezone_str = 'Australia/NSW'  # BoM uses Australian NSW time.
        my_logger.info("Setting Broken Hill tz to Australia/NSW")
    elif (np.abs(np.array([lng, lat]) - np.array([128.3, -25.033]
                                                 )
                 )).sum() < 0.1:  # Giles weather station which uses South Australian time
        timezone_str = 'Australia/South'  # BoM uses Australian Southern time.
        my_logger.info("Setting Giles Weather Stn to Australia/South")
    else:  # no specials
        pass

    timezone = pytz.timezone(timezone_str)

    # Get offsets from mid-winter of the current year to avoid any DST
    # if lat +ve then 21/12/current-year does not have daylight saving
    # If lat -ve then 21/6/current-year does not have daylight saving
    # TODO -- have info on date passed through to do this computation...
    if lat >= 0:
        no_dst_time = timezone.localize(datetime(datetime.now().year, 12, 21))
    else:
        no_dst_time = timezone.localize(datetime(datetime.now().year, 6, 21))

    # Calculate offsets in hours
    offset = no_dst_time.utcoffset()

    return offset


def read_gsdr_metadata(file: pathlib.Path) -> pd.Series:
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
     Time Zone:  CET, -> timezone
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
            if key == 'Other':  # end of header
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
    rename = {'New Units': 'units', 'Station ID': 'ID', 'Start datetime': 'Start', 'End datetime': 'End',
              'Time Zone': 'timezone', 'New Timestep': 'timestep'}
    for old_name, new_name in rename.items():
        result[new_name] = result.pop(old_name)
    # fix timezone...

    if result['timezone'].lower() == 'local standard time':
        delta_utc = utc_offset(lng=result['Longitude'], lat=result['Latitude'])
        my_logger.debug(f"UTC offset for local std time computed  from long/lat coords and is {delta_utc}")
        result["utc_offset"] = delta_utc
        result['Other'] = result['Other'] + " Changed LST"
    elif (result['timezone'].lower() == 'cet') and (
            result['ID'].startswith("AU_")):  # aus data with timezone CET is probable LST
        delta_utc = utc_offset(lng=result['Longitude'], lat=result['Latitude'])
        my_logger.debug(f"UTC offset for lAustralian CET computed  from long/lat coords and is {delta_utc}")
        result["utc_offset"] = delta_utc
        result['Other'] = result['Other'] + " Changed CET"
    else:  # raise a not-implemented error
        raise NotImplementedError(f"Implement UTC offset for general timezones {result['timezone']}")
    # make Start & End datetime
    start = result['Start']
    start = start[0:4] + "-" + start[4:6] + "-" + start[6:8] + "T" + start[8:]
    end = result['End']
    end = end[0:4] + "-" + end[4:6] + "-" + end[6:8] + "T" + end[8:]
    end = pd.to_datetime(end, yearfirst=True).tz_localize('UTC') - delta_utc
    start = pd.to_datetime(start, yearfirst=True).tz_localize('UTC') - delta_utc
    result['End_time'] = end
    result['Start_time'] = start

    result = pd.Series(result).rename(result['ID'])
    return result


def read_gsdr_data(meta_data: pd.Series) -> pd.Series:
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
                          periods=meta_data['Number of records']
                          )
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
    Read GSDR metadata saved to .csv file.
    :param file: file to read.
    :return: dataframe
    """
    df = pd.read_csv(file, index_col=0, parse_dates=['End_time', 'Start_time'])
    return df


def max_process(ds: xarray.Dataset, time_dim: str = 'time', minf_mean: float = 0.8) -> typing.Optional[xarray.Dataset]:
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
        'time_max_rain'
    )  # max times for this season

    base_name = max_rain.attrs['long_name']

    max_fraction.attrs['long_name'] = "Fraction present " + base_name
    max_fraction.attrs['units'] = '1'

    result = xarray.Dataset(
        dict(max_rain=max_rain, time_max_rain=max_time, mean_rain=mean_rain, max_fraction=max_fraction)
    )

    result.attrs.update(ds.attrs)
    my_logger.info(f"max computed for  {ds[time_dim][0].values} {memory_use()}")
    return result


def gen_mask(
        example_data_array: xarray.DataArray, radar_range: typing.Tuple[float, float] = (4.2e3, 144.8e3)
        # values empirically tuned.
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


def summary_process(
        ds: xarray.Dataset, mean_resample: str = '1h', time_dim: str = 'time', rain_threshold: float = 0.1,
        # From Pritchard et al, 2023 -- doi https://doi.org/10.1038/s41597-023-02238-4
        minf_mean: float = 0.8
) -> typing.Optional[xarray.Dataset]:
    f"""
    Process dataset for max (and mean & time of max).

    Args:
        ds: dataset to be processed. Must contain rainrate and isfile.
        mean_resample: the resample period to generate means.
        minf_mean: The minimum fraction data used to compute a time-mean. Data with a fraction below this are ignored when computing summaries.
        time_dim: the name of the time dimension
        rain_threshold: the threshold above which rain is used in *_thresh dataArrays



    Returns:dataset containing summary values. These are:
         max_rain, time_max_rain,  mean_rain 
          median_rain_thresh median for values > rain_threshold.
          mean_rate_thresh mean for values > rain_thresh
    Also includes some count values.
            f'count_{mean_resample}': number of mean non-missing values
            f'count_{mean_resample}_thresh': count of mean values > threshold
            f'samples_{mean_resample}': Total number of values
            rain_count: Count of all raw rain values above threshold
            rain_samples: Total number of possible samples.

    """

    def empty_ds(
            example_da: xarray.DataArray, non_time_variables: typing.Union[typing.List[str], str],
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
    count_of_mean = ok.astype('int64').sum(time_dim).load()  # count of mean samples with enough data.
    samples_of_mean = ok.count()  # count of mean samples
    if int(count_of_mean) == 0:  # no data..
        my_logger.warning(f"No data for {fraction[time_dim][[0, -1]].values}")
        result = empty_ds(rain.isel({time_dim: 0}, drop=True),
                          ["max_rain", "mean_rain", "median_rain_thresh", "mean_rain_thresh"], "time_max_rain"
                          )
        # need a count_rain_thresh which should be 0 (it must be less than count_of_mean
        count_rain_thresh = count_of_mean

    else:
        msk = (ds.isfile == 1)
        r = rain.fillna(0.0).where(msk)  # replace nans with 0 and apply mask.
        mean_rain_1h = r.resample({time_dim: mean_resample}).mean(skipna=True)
        my_logger.debug(
            f"mean computed using {mean_resample} for {mean_rain_1h[time_dim][[0, -1]].values} {memory_use()}"
        )
        mean_rain_1h = mean_rain_1h.sel({time_dim: ok}).load()  # select out where data OK
        my_logger.debug(f"rain computed for  {rain[time_dim].values[0]} {memory_use()}")
        bad = rain.isnull().all(time_dim, keep_attrs=True)  # find out where *all* data null
        if bad.all():
            my_logger.error(f"All missing at {bad[time_dim].isel[[0, -1]]}")
        # do the actual computation.
        m = mean_rain_1h > rain_threshold  # mask for rain threshold.
        rain_thresh = mean_rain_1h.where(m)
        count_rain_thresh = rain_thresh.count(time_dim)
        median_rain_thresh = rain_thresh.median(time_dim, keep_attrs=True, skipna=True).rename('median_rain_thresh')
        mean_rain_thresh = rain_thresh.mean(time_dim, keep_attrs=True, skipna=True).rename('mean_rain_thresh')
        max_rain = mean_rain_1h.max(time_dim, keep_attrs=True, skipna=True).rename('max_rain')
        mean_rain = mean_rain_1h.mean(time_dim, keep_attrs=True, skipna=True).rename('mean_rain')
        time_max_rain = mean_rain_1h.idxmax(time_dim, keep_attrs=False, skipna=True).rename('time_max_rain')
        result = xarray.Dataset(dict(median_rain_thresh=median_rain_thresh,
                                     mean_rain_thresh=mean_rain_thresh,
                                     max_rain=max_rain,
                                     mean_rain=mean_rain,
                                     time_max_rain=time_max_rain
                                     )
                                )

    # now have result either from empty case or std one,
    # now set the meta dat and fill in missing data with 0. Mild pain to do it here but want to be sure it
    # is the same regardless of how it made.
    base_name = ds.rainrate.attrs['long_name'] + f" mean {mean_resample}"
    variables = [v for v in result.variables if v.endswith("_rain")]
    for k in variables:
        if k.startswith("time_"):
            comp = k.split('_')[1]
            result[k].attrs['long_name'] = 'time of ' + comp + " " + base_name
        elif k.endswith("thresh"):
            comp = k.split('_')[0]
            result[k].attrs['long_name'] = comp + " " + base_name + f' for rain > {rain_threshold} mm/h'
            # result[k] = result[k].fillna(0.0) # fill in missing data with 0.0
        else:
            comp = k.split('_')[0]
            result[k].attrs['long_name'] = comp + " " + base_name
            # result[k] = result[k].fillna(0.0)
    # append min for median,
    count_rain_thresh.attrs['long_name'] = f'Count of rain {mean_resample} > {rain_threshold} mm/h'
    count_of_mean.attrs['long_name'] = f"Count of present {mean_resample} samples " + base_name
    samples_of_mean.attrs['long_name'] = f"{mean_resample} samples " + base_name
    counts_ds = xarray.Dataset(
        {
            f'count_{mean_resample}': count_of_mean,
            f'samples_{mean_resample}': samples_of_mean,
            "count_rain": rain_count,
            "samples_rain": rain_samples,
            f"count_{mean_resample}_thresh": count_rain_thresh
        }
    )
    for v in counts_ds.variables:
        counts_ds[v].attrs.pop('units', None)
        # would like count/sample vars to be ints so need to set to unsigned 64bit to have NaN.
        # danger is loss of precision.
        if v.startswith('count') or v.startswith('samples'):
            counts_ds[v] = counts_ds[v].astype('uint64')
    result = result.merge(counts_ds)
    result.attrs['threshold'] = rain_threshold
    result.attrs.update(ds.attrs)  # preserve initial metadata.
    my_logger.info(f"Summaries computed for  {ds[time_dim][[0, -1]].values} {memory_use()}")
    return result


def process_gsdr_record(
        gsdr_record: pd.Series,
        resample_max: str = 'QS-DEC'
) -> pd.DataFrame:
    """

    Args:
        gsdr_record: pandas series containing hourly rainfall
        resample_max: time period over which to resample

    Returns:Dataframe containing (as columns) indexed by time:
      #fraction -- fraction of non-missing data
      max -- value of max
      mean -- mean value
      time_max -- time max.

    """
    resamp = gsdr_record.resample(resample_max)
    count_data = (~gsdr_record.isnull()).resample(resample_max).count()
    fraction = (count_data / resamp.count()).rename('fraction')
    max_rain = resamp.max().rename("max_rain")

    def max_time_fn(series: pd.Series):
        if (~series.isnull()).sum() < 1:
            return np.nan
        else:
            return pd.Series.idxmax(series)

    time_max = resamp.apply(max_time_fn).rename("time_max_rain")
    # fix the type for time_max
    time_max = time_max.astype(gsdr_record.index.dtype)
    mean = resamp.mean().rename("mean_rain")  # need to convert to mm/season?
    total = (mean * count_data).rename('total_rain')
    df = pd.DataFrame([max_rain, mean, total, fraction]).T
    for s in ['max_rain', 'mean_rain', 'total_rain', 'fraction']:
        df[s] = df[s].astype('float')
    df['time_max_rain'] = max_rain.astype(gsdr_record.index.dtype)

    return df


def process_radar(
        data_set: xarray.Dataset,
        mean_resample: str = '1h',
        min_mean: float = 0.8,
        summary_resample: str = 'QS-DEC',
        time_dim: str = 'time',
        mask: typing.Optional[xarray.DataArray] = None,
        radar_range: typing.Tuple[float, float] = (4.2e3, 144.8e3)
) -> xarray.Dataset:
    """
    Compute various summaries of  rainfall on mean rainfall from  radar rainfall data.

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
                        minf_mean=min_mean
                        )

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
    Report on memory use.
    Returns: a string with the memory use in Gbytes or "Mem use unknown" if unable to load resource
     module.

    """
    try:
        import resource
        mem = f"Mem = {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6} Gbytes"
    except ModuleNotFoundError:
        mem = 'Mem use unknown'
    return mem


def summary_radar(
        files: typing.Union[typing.List[pathlib.Path], pathlib.Path],
        outfile: typing.Optional[pathlib.Path],
        mean_resample: str = '1h',
        summary_resample: str = 'MS',
        time_dim: str = 'time',
        chunks: typing.Optional[typing.Dict[str, int]] = None,
        overwrite: bool = False
) -> xarray.Dataset:
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
                       time_dim=time_dim
                       ).compute()

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
def init_log(
        log: logging.Logger,
        level: str,
        log_file: typing.Optional[typing.Union[pathlib.Path, str]] = None,
        mode: str = 'a'
):
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
                                  datefmt='%Y-%m-%d %H:%M:%S'
                                  )
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


typeDef = typing.Union[xarray.Dataset, np.ndarray]
typeDef = np.ndarray


def index_ll_pts(
        longitudes: typeDef,
        latitudes: typeDef,
        long_lats: typeDef,
        tolerance: float = 1e-3
) -> np.ndarray:
    """
    Extract long lat coords from an array
    Args:
        longitudes: numpy like array of longitudes
        latitudes: numpy like array of latitudes
        lon_lats: 2xN array of long/lats being searched for
        tolerance: Tolerance for match.
    Returns:2xN of row and column indices.
    """
    # check long_lats has shape 2,X
    if (long_lats.ndim != 2) or (long_lats.shape[0] != 2):
        raise ValueError("long_lats should be 2xN array")
    # code from ChatGPT3.5
    from scipy.spatial import KDTree
    # Flatten the 2D grid arrays and combine them into a single 2D array of coordinates
    grid_points = np.column_stack((longitudes.flatten(),
                                   latitudes.flatten())
                                  )

    # Construct a KD-Tree with the grid points
    tree = KDTree(grid_points)
    # Find the indices of the closest grid points for each query point
    distances, indices = tree.query(long_lats.T, workers=-1, distance_upper_bound=tolerance, p=1)
    # check distances are all less than tolerance and raise an error if not.
    if np.any(distances > tolerance):
        L = distances > tolerance
        raise ValueError(f"Some points too far away. {long_lats[:, L]}")

    # convert indices back to 2D indices
    row_indices, col_indices = np.unravel_index(indices, latitudes.shape)
    return np.column_stack((col_indices, row_indices)).T  # return as x,y


def lc_none(attrs: typing.Dict[str, str],
            key: str,
            default: typing.Optional[str] = None) -> typing.Optional[str]:
    """
    Handy function to retrieve value from attributes and lowercase it or return None if not present.
    :param attrs: Attributes dict
    :param key:kay
    :param default: default value to return if key not present.
    :return: value (or default if not found) lowercased
    """
    val = attrs.get(key, default)
    if val is None:  # not present
        return val
    return val.lower()  # lower case it.


def coarsen_ds(
        ds: xarray.Dataset,
        coarsen: typing.Dict[str, int],
        speckle_vars: typing.Tuple[str] = (),
        check_finite: bool = False,
        coarsen_method: typing.Literal['mean', 'median'] = 'mean',
        bounds_vars: typing.Tuple[str] = ('y_bounds', 'x_bounds'),
) -> xarray.Dataset:
    """
    Coarsen a dataset doing specials for dbz variables and bounds
    :param ds: dataset to be coarsened. All variables coarsened will be ordered.
    :param check_finite -- check all values are finite when taking exp and at end,
    :param coarsen: dict controlling coarsening
    :param coarsen_method -- how to coarsen. Either mean or median.
    :param bounds_vars: Tuple of variables that are bounds. They are coarsened by taking the min and max.
    :param speckle_vars: Tuple of variables that will have speckle computations done on them and included in the result.
    :return: coarsened dataset If speckle set will include variables with _speckle appended to the name.
    """
    if coarsen_method not in ['mean', 'median']:
        raise ValueError(f"Unknown coarsen method {coarsen_method}")
    my_logger.debug('Coarsening data')
    result = ds.copy().drop_vars(bounds_vars, errors='ignore')  # drop bounds vars as they need special processing,
    fail_vars = []
    vars_to_leave = [v for v in result.data_vars if v.endswith('_fract_high')]
    result = result.drop_vars(vars_to_leave, errors='ignore')  # drop fraction high vars
    for var in result.data_vars:
        unit = lc_none(result[var].attrs, 'units')
        if (unit is not None) and (unit == 'dbz'):
            fail_vars += [var]
    if len(fail_vars) > 0:
        raise ValueError(f'Vars {fail_vars} are still in dBZ. Converted to linear units before coarsening')

    # handle bounds vars
    bounds = dict()
    for var in bounds_vars:
        if var not in ds.data_vars:
            continue
        dim = var.split('_')[0]
        var_coarsen = {dim: coarsen[dim]}
        bounds[var] = xarray.concat(
            [ds[var].isel(n2=0).coarsen(var_coarsen).min(),
             ds[var].isel(n2=1).coarsen(var_coarsen).max()], 'bounds'
        ).T
        my_logger.debug('Coarsened bounds for ' + var)

    coarse = result.coarsen(**coarsen)
    if coarsen_method == 'mean':
        result = coarse.mean()  # coarsen whole dataset
    elif coarsen_method == 'median':
        result = coarse.median()
    else:
        raise ValueError(f"Unknown coarsen method {coarsen_method}")
    for var in  speckle_vars:
        speckle= ds[var].coarsen(**coarsen).std()
        speckle = speckle.rename(var+'_speckle')
        result = result.merge(speckle)
    result = result.merge(xarray.Dataset(bounds))  # overwrite bounds.
    for v in vars_to_leave:
        result[v] = ds[v]
    if check_finite:
        fail = False
        for var in result.data_vars:
            if np.isinf(result[var].values).any():
                my_logger.warning(f'Inf values in {var}')
                fail = True

        if fail:
            raise ValueError('Non-finite values in coarsened data')
    my_logger.debug('Coarsened data')
    return result


def read_radar_zipfile(
        path: pathlib.Path,
        concat_dim: str = 'valid_time',
        first_file: bool = False,
        region:typing.Optional[typing.Dict[str,slice]] = None,
        **load_kwargs
) -> typing.Optional[xarray.Dataset]:
    """
    Read netcdf data from a zipfile containing lots of netcdf files
    Args:
        :param path: Path to zipfile
        :param concat_dim: dimension to concatenate along
        :param bounds_vars: tuple of variables that are bounds
      :param first_file: If True only read in the first file
      :param region -- region to extract data from
    :param **load_kwargs: kwargs to be passed to xarray.open_mfdataset
    Returns: xarray dataset or None if nothing successfully read.

    """

    my_logger.debug(f'Unzipping and reading in data {memory_use()} for {path}')
    with tempfile.TemporaryDirectory() as tdir:
        # times on linux workstation
        zipfile.ZipFile(path).extractall(tdir)  # extract to temp dir 0.16 seconds
        files = list(pathlib.Path(tdir).glob('*.nc'))  #  0.0 seconds
        if first_file:
            files = files[0:1]
        if load_kwargs.get('combine') == 'nested':
            load_kwargs.update(concat_dim=concat_dim)
        try:
            ds = xarray.open_mfdataset(files, **load_kwargs)

        except RuntimeError as error:
            my_logger.warning(
                f"Error reading in {files[0]} - {files[-1]} {error} {type(error)}. Trying again loading individual files")
            time.sleep(0.1)  # sleep a bit and try again with loading
            datasets = []
            load_args = load_kwargs.copy()
            # drop things that don't work with load!
            for key in ['combine', 'concat_dim', 'parallel']:
                if key in load_args:
                    load_args.pop(key)
            for file in files:  # loop over files loading each one. If get a failure complain and skip.
                try:
                    ds = xarray.open_dataset(file, **load_args)
                    datasets.append(ds)
                except Exception as e:
                    my_logger.warning(f"Error reading in {file} {e} -- skipping")
            if len(datasets) == 0:
                return None
            ds = xarray.concat(datasets, concat_dim)  #
        # check have not got zero len dims on region.
        if region is not None:
            ds = ds.sel(**region)
            bad_dims = [rname for rname in region.keys() if ds.sizes[rname] == 0]
            if len(bad_dims) > 0:
                raise ValueError('Following dimensions have zero length: ' + ','.join(bad_dims))

        if first_file:
            ds = ds.drop_vars(concat_dim, errors='ignore')  # drop the concat dim as only want the first value
        # check dimension sizes are OK


        ds = ds.compute()  # compute/load before closing the files.
        ds.close()  # close the files
        my_logger.debug('Read in data. Cleaning tempdir')
    # fix y_bounds which is in reverse order from x_bounds.
    var = 'y_bounds'
    if var in ds.data_vars:
        v = ds[var]
        ds[var] = xarray.concat([v.isel(n2=1).assign_coords(n2=0), v.isel(n2=0).assign_coords(n2=1)], dim='n2')

    if not first_file:
        my_logger.debug(f'read in {len(ds[concat_dim])} times from {path} {memory_use()}')
    return ds


def site_info(site: str) -> pd.DataFrame:
    """
    Get information on a site.
    Args:
        site: site to get info on
    Returns: pandas series with information on site.
    """
    from importlib.resources import files
    file = files('meta_data') / 'long_radar_stns.csv'
    df = pd.read_csv(file, index_col='id_long').drop(columns='Unnamed: 0')
    L = df.short_name == site
    meta = df[L]
    return meta


def read_acorn(
        site: int,
        retrieve: bool = False,
        what: typing.Literal['max', 'min', 'mean'] = 'mean'
) -> pd.Series:
    """
    Read in ACORN temperature data for a site.
    Args:

        site: site to read in
        what: What to retrieve (max, min or mean). If mean max and min will be retrieved and averaged
        retrieve: If True ignore local cache.


    Returns: pandas series
    """

    if what == 'mean':
        my_logger.debug('Computing mean from avg of max and min')
        minT = read_acorn(site, retrieve=retrieve, what='min')
        maxT = read_acorn(site, retrieve=retrieve, what='max')
        mean = ((minT + maxT) / 2.0).rename('mean')
        mean.attrs.update(var='mean temperature (degC)')
        return mean
    cache_dir = data_dir / 'acorn_data'
    cache_dir.mkdir(exist_ok=True, parents=True)
    filename = cache_dir / f'{site:06d}_{what}.csv'

    #  retrieve the data if needed
    if retrieve or (not filename.exists()):
        url = f'http://www.bom.gov.au/climate/change/hqsites/data/temp/t{what}.{site:06d}.daily.csv'
        my_logger.info(f"Retrieving data from {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:77.0) Gecko/20190101 Firefox/77.0'}
        # fake we are interactive...
        # retrieve data writing it to local cache.
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    df = pd.read_csv(filename, index_col='date', parse_dates=True)
    # extract the site number and name from the dataframe
    site_number = df.iloc[0, 1]
    site_name = df.iloc[0, 2]
    df = df.drop(index='NaT').iloc[:, 0]
    df.attrs.update(site_number=site_number, site_name=site_name, var=df.name)
    df = df.rename(what)
    return df

def list_vars(data_set:xarray.Dataset,prefix:str,
              show_dims:bool = False,
              show_attrs:bool = False,
              fract_nan:bool = False) -> typing.List[str]:
    """
    List variables in a dataset with a given prefix
    Args:
        data_set: dataset
        prefix: prefix to search for
        show_dims: If True print out dimensions
        show_attrs: If True print out attributes
        fract_nan: If True show fraction of nan values
        Returns: list of variables with prefix
    """
    vars = [str(v) for v in data_set.data_vars if prefix in str(v)]
    for v in vars:
        sprint_str= ''
        if show_dims:
            sprint_str += f'{data_set[v].dims}'
        if show_attrs:
            sprint_str += f'{data_set[v].attrs}'
        if fract_nan:
            sprint_str += f'Fraction of NaNs {data_set[v].isnull().mean().values:.4f}'
        if len(sprint_str) > 0:
            print(f'{v} {sprint_str}')
    return vars
