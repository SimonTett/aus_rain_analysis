# Library for australian analysis.
from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import pathlib
import platform
import sys
import tempfile
import time
import zipfile
import cartopy.crs as ccrs
import cartopy.geodesic
import matplotlib
import matplotlib.pyplot as plt

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
import ast  # thanks chaptGPT co-pilot
import numpy.random as random

# dict of site names and numbers.
site_numbers = dict(Adelaide=46, Melbourne=2, Wtakone=52, Sydney=3, Brisbane=50, Canberra=40,
                    Cairns=19, Mornington=36, Grafton=28, Newcastle=4, Gladstone=23
                    )
site_names = dict(zip(site_numbers.values(), site_numbers.keys()))  # reverse lookup
region_names = dict(Tropics=['Cairns', 'Mornington'],
                    QLD=['Gladstone', 'Brisbane', 'Grafton'],
                    NSW=['Newcastle', 'Sydney', 'Canberra'],
                    South=['Melbourne', 'Wtakone', 'Adelaide']
                    )
# need a reverse sort -- sites -> regions
names_to_region = dict()
for site in site_numbers.keys():
    for region, sites in region_names.items():
        if site in sites:
            names_to_region[site] = region
            break
acorn_lookup = dict(Adelaide=23000, Melbourne=86338, Wtakone=96003, Sydney=66214, Brisbane=40842, Canberra=70351,
                    Cairns=31011, Mornington=29077,
                    Grafton=59151, Newcastle=61078, Gladstone=39083
                    )  # acorn station IDs for each site.
hostname = platform.node()
if hostname.startswith('gadi'):  # aus super-computer
    radar_dir = pathlib.Path("/g/data/rq0/level_2/")
    data_dir = pathlib.Path("/scratch/wq02/st7295/radar/")
    hist_ref_dir = pathlib.Path("/g/data/rq0/hist_gndrefl/")
elif hostname.startswith('ccrc'):  # CCRC desktop
    data_dir = pathlib.Path("/home/z3542688/OneDrive/data/aus_radar_analysis/radar")
    common_data = pathlib.Path("/home/z3542688/OneDrive/data/common_data")
elif hostname == 'geos-w-048':  # my laptop
    data_dir = pathlib.Path(r"C:\Users\stett2\OneDrive - University of Edinburgh\data\aus_radar_analysis\radar")
    common_data = pathlib.Path(r"C:\Users\stett2\OneDrive - University of Edinburgh\data\common_data")
else:
    raise NotImplementedError(f"Do not know where directories are for this machine:{hostname}")
data_dir.mkdir(exist_ok=True, parents=True)  # make it if need be!
module_path = pathlib.Path(__file__).parent  # path to this module

my_logger = logging.getLogger(__name__)  # for logging
# dict to control logging


timezone_finder = TimezoneFinder()  # instance of timezone_finder -- only want one,


def extract_rgn(radar_ds: xarray.Dataset) -> typing.Dict[str, slice]:
    """
    Extract region from radar dataset attributes.
    Args:
        radar_ds: radar dataset

    Returns: dictionary with region.
    """
    # extract the region used from the meta-data though need to reverse  y
    rgn = [v for v in radar_ds.attrs['program_args'] if v.startswith('region:')][0]
    rgn = np.array(ast.literal_eval(rgn.split(':')[1]))  # thanks chatgpt fot this
    # convert to m from km
    rgn *= 1000.
    rgn = dict(x=slice(*rgn[0:2]), y=slice(*rgn[-1:-3:-1]))  # need to reverse the y index.

    return rgn


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


def utc_offset(lng: float = 0.0, lat: float = 50.0) -> (timedelta, str):
    """
    Work out the standard time offset from UTC for specified lon/lat co-ord.
    TODO: consider passing in the times so can deal with changing timezones for location.
    Args:
        lng: longitude (in decimal degrees) of point
        lat: latitude (in decimal degrees) of point

    Returns: timedelta from UTC and timezone string
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

    return offset, timezone_str


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
        delta_utc, tz = utc_offset(lng=result['Longitude'], lat=result['Latitude'])
        my_logger.debug(f"UTC offset for local std time computed  from long/lat coords and is {delta_utc}")
        result["utc_offset"] = delta_utc
        result['Other'] = result['Other'] + " Changed LST"
        result['Inferred_timezone'] = tz
    elif (result['timezone'].lower() == 'cet') and (
            result['ID'].startswith("AU_")):  # aus data with timezone CET is probable LST
        delta_utc, tz = utc_offset(lng=result['Longitude'], lat=result['Latitude'])
        my_logger.debug(f"UTC offset for Australian CET computed  from long/lat coords and is {delta_utc}")
        result["utc_offset"] = delta_utc
        result['Inferred_timezone'] = tz
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

    mean = resamp.mean().rename("mean_rain")  # need to convert to mm/season?
    total = (mean * count_data).rename('total_rain')
    df = pd.DataFrame([max_rain, mean, total, fraction]).T
    for s in ['max_rain', 'mean_rain', 'total_rain', 'fraction']:
        df[s] = df[s].astype('float')
    # fix the type for time_max
    time_max = time_max.astype(gsdr_record.index.dtype)
    df['time_max_rain'] = time_max.astype(gsdr_record.index.dtype)

    return df


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
        datefmt: typing.Optional[str] = '%Y-%m-%d %H:%M:%S',
        mode: str = 'a'
):
    """
    Set up logging on a logger! Will clear any existing logging
    :param log: logger to be changed
    :param level: level to be set.
    :param log_file:  if provided pathlib.Path to log to file
    :param mode: mode to open log file with (a  -- append or w -- write)
    :param datefmt: date format for log.
    :return: nothing -- existing log is modified.
    """
    log.handlers.clear()
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:  %(message)s',
                                  datefmt=datefmt
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


def lc_none(
        attrs: typing.Dict[str, str],
        key: str,
        default: typing.Optional[str] = None
) -> typing.Optional[str]:
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
    for var in speckle_vars:
        speckle = ds[var].coarsen(**coarsen).std()
        speckle = speckle.rename(var + '_speckle')
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
        region: typing.Optional[typing.Dict[str, slice]] = None,
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
                f"Error reading in {files[0]} - {files[-1]} {error} {type(error)}. Trying again loading individual files"
            )
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


def read_radar_file(path: pathlib.Path | str) -> pd.DataFrame:
    """
    Read in radar data from a file.
    Args:
        :param path -- path to csv file
    """
    df = pd.read_csv(path, index_col='id_long', parse_dates=['postchange_start', 'prechange_end'],
                     dayfirst=True, na_values='-',
                     true_values=['Yes'],
                     dtype={'site_lat': np.float32, 'site_lon': np.float32, 'site_alt': np.float32}
                     )

    # Set NaN to False
    for col in ['dp', 'doppler']:
        df[col] = ~df[col].isnull()
    return df


def site_info(site_no: int) -> pd.DataFrame:
    """
    Get information on a site.
    Args:
        site_no: site to get info on
    Returns: pandas series with information on site.
    """
    from importlib.resources import files
    file = files('meta_data') / 'long_radar_stns.csv'
    df = read_radar_file(file)
    L = df.id == site_no
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


def list_vars(
        data_set: xarray.Dataset, prefix: str,
        show_dims: bool = False,
        show_attrs: bool = False,
        fract_nan: bool = False
) -> typing.List[str]:
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
        sprint_str = ''
        if show_dims:
            sprint_str += f'{data_set[v].dims}'
        if show_attrs:
            sprint_str += f'{data_set[v].attrs}'
        if fract_nan:
            sprint_str += f'Fraction of NaNs {data_set[v].isnull().mean().values:.4f}'
        if len(sprint_str) > 0:
            print(f'{v} {sprint_str}')
    return vars


def gen_radar_projection(longitude: float, latitude: float, parallel_offset: float = 1.5) -> dict:
    proc_attrs = {
            "grid_mapping_name"            : "albers_conical_equal_area",
            "standard_parallel"            : np.round([latitude - parallel_offset, latitude + parallel_offset], 1),
            "longitude_of_central_meridian": longitude,
            "latitude_of_projection_origin": latitude,
            "false_easting"                : 0.0,
            "false_northing"               : 0.0,
            "semi_major_axis"              : 6378137.0,
            "semi_minor_axis"              : 6356752.31414
    }
    return proc_attrs


def radar_projection(attrs: dict) -> ccrs.Projection:
    """
    Create a projection from radar projection information.
    :param attrs: attributes containing projection information.
    :return: cartopy projection
    """

    kw_proj = attrs.copy()
    if kw_proj.pop('grid_mapping_name') != 'albers_conical_equal_area':
        raise ValueError("Only Albers Equal Area supported")
    globe = ccrs.Globe(semimajor_axis=kw_proj.pop('semi_major_axis'),
                       semiminor_axis=kw_proj.pop('semi_minor_axis')
                       )  # globe used.
    # rename some keys
    kw_proj['central_longitude'] = kw_proj.pop('longitude_of_central_meridian')
    kw_proj['central_latitude'] = kw_proj.pop('latitude_of_projection_origin')
    kw_proj['standard_parallels'] = kw_proj.pop('standard_parallel').tolist()
    proj_radar = ccrs.AlbersEqualArea(globe=globe, **kw_proj)

    return proj_radar


def add_long_lat_coords(data_set: xarray.Dataset) -> xarray.Dataset:
    """
    Add long/lat coords to a data array.
    Must contain a proj variable which is used to compute the projection. From that
    the long/lat coords are computed.
    Args:
        data_set:   dataset to add long/lat coords to.
    Returns: dataset with long/lat coords added
    """
    proj = radar_projection(data_set.proj.attrs)
    # get the lon & lat coords for the radar
    X, Y = np.meshgrid(data_set.x.values, data_set.y.values)
    coords = ccrs.PlateCarree().transform_points(src_crs=proj, x=X, y=Y)
    vars_xy = [v for v in data_set.data_vars if
               ('x' in data_set[v].dims) and ('y' in data_set[v].dims)]
    result = data_set.copy()
    for v in vars_xy:
        result[v] = result[v].assign_coords(longitude=(['x', 'y'], coords[:, :, 0]),
                                            latitude=(['x', 'y'], coords[:, :, 1])
                                            )

    return result


def data_array_station_match(
        data_array: xarray.DataArray,
        projection: ccrs.Projection,
        station_metadata: pd.DataFrame,
        tolerance: typing.Optional[float] = None,
        array_coords: typing.Tuple[str, str] = ('x', 'y'),
        station_coords: typing.Tuple[str, str] = ('Longitude', 'Latitude')
) -> xarray.DataArray:
    """
    Extract radar dataset to the gauge locations.

    :param data_array: radar datat array to be extracted
    :param projection -- projection of the  data
    :param station_metadata:  metadata for station. Only Longitude & Latitude are used. Index used to name the data.
    :param tolerance: tolerance in units of projection for matching the data_array to the station cooords.
    :param array_coords: tuple of the names of the x and y coords in the data_array.
    :param station_coords: tuple of the names of the Longitude and Latitude in the station_metadata.
    :return: Data array of  data at the station locations
    """
    # work out the co-ords of the stations  in the  projection for the data_array
    station_coords = projection.transform_points(src_crs=ccrs.PlateCarree(),
                                                 x=station_metadata.loc[:, station_coords[0]],
                                                 y=station_metadata.loc[:, station_coords[1]]
                                                 )
    # construct the selector so can use "fancy" indexing.
    sel = dict()
    for ind, c in enumerate(array_coords):
        sel[c] = xarray.DataArray(station_coords[:, ind],
                                  dims='station', coords=dict(station=station_metadata.index)
                                  )
    match = data_array.sel(method='nearest', tolerance=tolerance, **sel)  # do the match

    return match


type_time_range = typing.Tuple[typing.Optional[str], typing.Optional[str]]


def read_gauge_metadata(
        radar_coords: np.ndarray,
        radius: float,
        time_range: type_time_range = (None, None)
) -> pd.DataFrame:
    """
    Read the gauge metadata file, extract the gauges within the radius of the radar
    :param radar_coords: coords as numpy array (longitude, latitude) of the radar
    :param radius: radius in m
    :param time_range: time range to extract the gauges.
        Extracted if the start date is greater than the first element and the end date is less than the second element.
    :return: pd.DataFrame
    """
    #TODO -- when ausLib has a root directory use that so can run this code from anywhere.
    all_metadata = read_gsdr_csv("meta_data/AU_GSDR_metadata.csv")
    earth_geo = cartopy.geodesic.Geodesic()
    pts = np.array([all_metadata.Longitude, all_metadata.Latitude]).T
    dist = earth_geo.inverse(radar_coords, pts)
    L = (dist[:, 0] < radius)
    with pd.option_context("mode.copy_on_write", True):
        radar_metadata = all_metadata[L]  # extract to  points close enough
        radar_metadata['Distance'] = dist[L, 0]
    if time_range[0] is not None or time_range[1] is not None:
        # filter on time.
        L = pd.Series(True, index=radar_metadata.index)
        if time_range[0] is not None:
            L = L & (radar_metadata['End_time'] >= time_range[0])  # Finish *after* the desired start date
        if time_range[1] is not None:
            L = L & (radar_metadata['Start_time'] <= time_range[1])  # Start *before* the desired end date
        radar_metadata = radar_metadata[L]
    return radar_metadata


def add_std_arguments(parser: argparse.ArgumentParser, dask: bool = True) -> None:
    """
    :param parser: parser to be modified.
    :param dask: If True add dask argument
    Add std arguments to a parser.
    These are:
    -v -- verbose
    --dask -- turn on dask. (if dask set)
    --log_file -- have a log file.
    --overwrite -- overwrite files.
    And  arguments for submitting jobs to a queue system.
    --submit -- submit to queue system
    --queue -- queue to submit to
    --project -- project to submit to
    --storage -- storage options
    --time -- time to run for
    --cpus -- number of cpus to use.
    --memory -- memory to use
    --job_name -- job name
    --email -- email address to send job info to
    --log_base -- base name for q system logs
    --json_submit_file -- JSON file containing default options.
                   These are overwritten by command line args.
                   Keys are the same as the long form of the argument with -- removed.
                   Any keys ending in comment are also removed

    --setup_script -- script to run before running the command
    --holdafter -- hold job until the held after job has ran.
    --dryrun -- dry run only. Nothing will be submitted and script will exit
    --history_file -- store command in history file
    --purpose -- purpose of the job. Does nothing but put in log file
    Returns: nada

    """
    std = parser.add_argument_group('Standard')
    std.add_argument('--overwrite', action='store_true', help='Overwrite files')
    std.add_argument('-v', '--verbose', action='count', help='Verbose output', default=0)
    if dask:  # some codes don't want dask.
        std.add_argument('--dask', action='store_true', help='Start dask client')
    std.add_argument('--log_file',
                     help='Name of log file -- if provided. log info goes there as well as std out/err'
                     )

    # add a submit sub-command,
    # note that a value of '' means do not do anything. This is to allow checking for values we actually need
    submit = parser.add_argument_group('Submit. Only used if --submit set')

    submit.add_argument('--submit', help='Submit self to Q system', action='store_true')
    submit.add_argument('--queue', help='Queue to submit to')
    submit.add_argument('--project', help='Project to submit to')
    submit.add_argument('--storage', help='Storage options')
    submit.add_argument('--time', help='Time to run for', default='1:00:00')
    submit.add_argument('--cpus', type=int, help='Number of nodes', default=1)
    submit.add_argument('--memory', help='Memory to use', default='4G')
    submit.add_argument('--job_name', help='Job name')
    submit.add_argument('--email', help='Email address to send job info to', default='')
    submit.add_argument('--log_base', type=pathlib.Path, help='Base name for q system logs')
    submit.add_argument('--json_submit_file', type=pathlib.Path,
                        help='JSON file containing default options. These overwritten by command line args'
                        )
    submit.add_argument('--setup_script', help='Script to run before running the command', type=pathlib.Path)
    submit.add_argument('--holdafter', help='Hold job until the held after job(s) have ran.', nargs='+')
    submit.add_argument('--dryrun', help='Dry run only. Nothing will be submitted and script will exit',
                        action='store_true'
                        )
    submit.add_argument('--history_file', help='Store command in history file', type=pathlib.Path)
    submit.add_argument('--purpose', nargs='+', help='Purpose of the job. Does nothing but put in log file')


def gen_pbs_script(
        cmd_to_run: str,
        holdafter: typing.Optional[list[str]] = None,
        queue: typing.Optional[str] = None,
        project: typing.Optional[str] = None,
        storage: typing.Optional[str] = None,
        time: typing.Optional[str] = None,
        cpus: int = 1,
        memory: typing.Optional[str] = None,
        job_name: typing.Optional[str] = None,
        email: str = '',
        log_base: typing.Optional[pathlib.Path] = None,
        setup_script: typing.Optional[pathlib.Path] = None,
        ) -> tuple[list[str], str]:
    # generate the qsub command
    if holdafter is None:
        holdafter = []  # set it to empty list.
    # check all values are set.
    args = locals()
    none_vars = []

    for name, value in args.items():
        if value is None:
            my_logger.warning(f'Must set {name}')
            none_vars += [name]
    if len(none_vars) > 0:
        raise ValueError(f'Must set {" ".join(none_vars)} to values')
    qsub_cmd = ["qsub"]
    wd = pathlib.Path.cwd()  # where we are now.
    qsub_cmd.append('-')  # read from std in.
    my_logger.debug('qsub_cmd: ' + ' '.join(qsub_cmd))
    if len(job_name) > 15:
        job_name = job_name[0:14]  # must be less than 15 chars.
        my_logger.debug('Truncated job_name to 15 chars')
    # make the directory for logs
    log_base.parent.mkdir(exist_ok=True, parents=True)
    # check setup script exists
    if not setup_script.exists():
        raise FileNotFoundError(f'Setup script {setup_script} does not exist')
    # construct the PBS script
    log_stdout = log_base.with_name(log_base.name + '.out')
    log_stderr = log_base.with_name(log_base.name + '.err')
    cmd_str = f"""
#PBS -P {project}
#PBS -q {queue}
#PBS -l walltime={time}
#PBS -l storage={storage}
#PBS -l mem={memory}
#PBS -l ncpus={cpus}
#PBS -l jobfs=20GB
#PBS -l wd
#PBS -N {job_name}
#PBS -o {log_stdout}
#PBS -e {log_stderr}
"""
    if (len(email) > 0) and ('@' in email):
        cmd_str += f"""
#PBS -m e
#PBS -M {email}
    """
    if len(holdafter) > 0:
        cmd_str += f'#PBS -W depend=afterok:{":".join(holdafter)}\n'
    cmd_str += fr'''
export TMPDIR=$PBS_JOBFS
cd {wd} || exit # make sure we are in the right directory        
. {setup_script} # setup software and then run the command
cmd="{cmd_to_run}"
echo "Cmd is $cmd"
result=$($cmd)
echo $result
'''
    my_logger.debug('Cmd_str: ' + cmd_str)
    return qsub_cmd, cmd_str


def process_std_arguments(args: argparse.Namespace) -> logging.Logger:
    """
    Process standard arguments. See add_std_arguments for setup.
    Args:
        args: arguments to be processed.
          Deals with verbose, log_file and dask and submission options

    Returns: logger
    """
    #raise NotImplementedError('Needs some testing!')
    my_logger = setup_log(args.verbose, args.log_file)
    # log the args
    for name, value in vars(args).items():
        my_logger.info(f"Arg:{name} =  {value}")
    # deal with submission.
    if args.submit:
        my_logger.debug('Submitting job')
        cmd = [c for c in sys.argv if not (c.startswith('--submit') or c.startswith('--dryrun'))]
        cmd = ' '.join(cmd)  # remake the cmd.
        my_logger.info('Cmd is: ' + cmd)

        # generate the command and qsub_cmd
        sub_args = dict(project=args.project, queue=args.queue, storage=args.storage,
                        time=args.time, cpus=args.cpus, memory=args.memory, job_name=args.job_name,
                        email=args.email, log_base=args.log_base
                        )
        history_file = args.history_file
        if args.json_submit_file:
            with open(args.json_submit_file) as f:
                json_args = json.load(f)
            # remove anything with a comment in it.
            keys = list(json_args.keys())  # need to get out keys as removing them
            for k in keys:
                if k.endswith('comment'):
                    del json_args[k]
                    my_logger.debug(f'removed {k} from json file')
            # convert things that should be paths from strings to paths
            keys = ['setup_script', 'log_base', 'history_file']  # things that should be converted to paths
            for k in keys:
                try:
                    json_args[k] = pathlib.Path(json_args[k])
                except KeyError:
                    pass
            # deal with history
            try:
                history_file = json_args.pop('history_file')
            except KeyError:
                pass  # no history file

            my_logger.debug(f'Updating {json_args} with sub_args')

            sub_args.update({k: v for k, v in json_args.items() if (k in sub_args.keys() and
                                                                    (sub_args.get(k) is not None) or (sub_args.get(
                        k
                    ) != ''))}
                            )  # update not None/empty

            if sub_args['cpus'] > 1 and not args.dask:
                my_logger.warning('Setting cpus to 1 as not using dask. Waiting for 5 seconds.')
                sub_args['cpus'] = 1
                time.sleep(5)  # so can see the warning

        qsub_cmd, cmd_str = gen_pbs_script(cmd, holdafter=args.holdafter, **sub_args)
        if not args.dryrun:  # actually run the job
            my_logger.info('Running qsub: ' + ' '.join(qsub_cmd))
            # if we have a history file append to it.
            if history_file is not None:
                history_file.parent.mkdir(exist_ok=True, parents=True)  # make directory if needed
                with open(history_file, 'a') as f:
                    print(str(datetime.utcnow()) + ': ' + cmd, file=f)
            import subprocess
            result = subprocess.run(qsub_cmd, input=cmd_str, text=True, capture_output=True)
            if result.returncode != 0:
                my_logger.warning(f'Error submitting job stdout: {result.stdout}')
                my_logger.warning(f'Error submitting job stderr: {result.stderr}')
            result.check_returncode()
            my_logger.warning(f'Output from {qsub_cmd} is {result.stdout}')
            if len(result.stderr) > 0:
                my_logger.warning(f'Stderr from {qsub_cmd} is {result.stderr}')
            print(result.stdout)  # so can get the job id.
        else:
            print("fake_job.999999")
        my_logger.debug('Submitted so exiting')
        sys.exit(0)

    try:
        if args.dask:
            dask_client()
    except AttributeError:  # no dask
        my_logger.debug('No dask setup')
    return my_logger


def setup_log(
        verbose_level: int,
        log_file: typing.Optional[str] = None
) -> logging.Logger:
    """
    Set up logging
    Args:
        verbose_level: level of verbosity
        log_file: log file to write to if not None

    Returns: logger
    """
    if verbose_level > 1:
        level = 'DEBUG'
    elif verbose_level > 0:
        level = 'INFO'
    else:
        level = 'WARNING'
    init_log(my_logger, level=level, log_file=log_file, mode='w')
    return my_logger


def sample_events(
        data_set: xarray.Dataset,
        nsamples: int = 100,
        rng: np.random.BitGenerator = None,
        dim: str = 'EventTime',
        dim2: str = 'quantv'
) -> xarray.Dataset:
    """
    Sample events with replacement from a data array
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
    ds = data_set.dropna(dim)
    npts = ds.EventTime.size
    locs = rng.integers(0, npts, size=(nsamples, npts))  # boot strap samples.
    # wrap it up in  data array to allow fancy indexing
    coords = dict(bootstrap_sample=np.arange(0, nsamples), index=np.arange(0, npts))
    locs = xarray.DataArray(locs, coords=coords)
    ds = ds.isel({dim: locs})  # and sample at dim
    # pick random dim2 samples for each case and collapse
    rand_index = rng.integers(1, len(data_set[dim2]) - 2, size=(nsamples, npts))  # don't want the min or max quantiles
    rand_index = xarray.DataArray(rand_index, coords=(ds.bootstrap_sample, ds.index))
    ds = ds.isel({dim2: rand_index}).drop_vars([dim, dim2])
    return ds


def sample_events2(
        data_set: xarray.Dataset,
        nsamples: int = 100,
        rng: np.random.BitGenerator = None,
        dim: str = 'EventTime',

) -> xarray.Dataset:
    """
    Sample events with replacement from a data array
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
    ds = data_set.dropna(dim)
    npts = ds.EventTime.size
    locs = rng.integers(0, npts, size=(nsamples, npts))  # boot strap samples.
    # wrap it up in  data array to allow fancy indexing
    coords = dict(bootstrap_sample=np.arange(0, nsamples), index=np.arange(0, npts))
    locs = xarray.DataArray(locs, coords=coords)
    ds = ds.isel({dim: locs})  # and sample at dim

    return ds


def comp_radar_fit(
        dataset: xarray.Dataset,
        cov: typing.Optional[list[str] | str] = None,
        n_samples: int = 100,
        bootstrap_samples: int = 0,
        rng_seed: int = 123456,
        file: typing.Optional[pathlib.Path] = None,
        bootstrap_file: typing.Optional[pathlib.Path] = None,
        recreate_fit: bool = False,
        name: typing.Optional[str] = None,
        extra_attrs: typing.Optional[dict] = None,
        use_dask: bool = False
) -> tuple[xarray.Dataset, typing.Optional[xarray.Dataset]]:
    """
    Compute the GEV fits for radar data. This is a wrapper around the R code to do the fits.
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
    ds = dataset.drop_vars('Observed_temperature', errors='ignore').isel(quantv=rand_index). \
        rename(dict(quantv='sample')).assign_coords(**coord)
    if use_dask:
        my_logger.debug(f'Chunking data for best est  {memory_use()}')
        samp_chunk = max(n_samples // 10, 1)
        ds = ds.chunk(sample=samp_chunk, resample_prd=1, EventTime=-1)  # parallelize over sample
        print(ds.chunksizes)
        #ds = ds.compute()
        my_logger.debug(f'Rechunked data for best est  {memory_use()}')
    cov_rand = None
    if cov is not None:
        cov_rand = [ds[c] for c in cov]

    wt = ds.count_cells
    mx = ds.max_value
    my_logger.debug('Doing GEV fit')
    fit = gev_r.xarray_gev(mx, cov=cov_rand, dim='EventTime', weights=wt, verbose=True,
                           recreate_fit=recreate_fit, file=file, name=name, extra_attrs=extra_attrs,
                           use_dask=use_dask
                           )
    my_logger.debug('Done GEV fit')

    if bootstrap_samples > 0:
        my_logger.info(f"Calculating bootstrap for {name} {memory_use()}")
        # loop over bootstrap samples and then concat together at the end.
        # generating all the samples produces a very large mem object and requires data
        # to be moved around. So don't bother...

        ds_bs = ds.groupby('resample_prd').map(sample_events2,
                                               rng=rng, nsamples=bootstrap_samples,
                                               dim='EventTime'
                                               ).drop_vars('EventTime')
        if use_dask:
            # chunking. Assuming running on small number of cores.  Say 10.
            # so we want min number of chucks to give abotu 10.
            bs_samp = max(bootstrap_samples // 10, 1)
            samp_chunk = max(n_samples // 10, 1)
            ds_bs = ds_bs.unify_chunks().chunk(
                sample=samp_chunk, bootstrap_sample=-1,
                resample_prd=1, index=-1
            )  # parallelize and needs tuning
            my_logger.debug(f'Rechunked data for bs calcs {memory_use()}')

        cov_rand = None
        if cov is not None:
            cov_rand = [ds_bs[c] for c in cov]
        my_logger.debug('Doing BS GEV fit')
        bs_fit = gev_r.xarray_gev(ds_bs.max_value, cov=cov_rand, dim='index', weights=ds_bs.count_cells, verbose=True,
                                  recreate_fit=recreate_fit, file=bootstrap_file, name=name + '_bs',
                                  extra_attrs=extra_attrs, use_dask=use_dask
                                  ).mean('sample')
        my_logger.debug(f"Computed Bootstrap fits {memory_use()}")

    else:
        bs_fit = None

    return fit, bs_fit


def write_out(
        data: xarray.Dataset | xarray.DataArray,
        outpath: pathlib.Path,
        time_unit: typing.Optional[str] = None,
        extra_attrs: typing.Optional[dict] = None,
        time_dim: typing.Optional[str] = 'time'
) -> None:
    """
    Write out data. Encoding and attributes are modified.
    :param data: data array or dataset to be written out
    :param time_unit: units of time.
    :param outpath: path for where data to be written to
    :param extra_attrs: extra attributes for the dataset
    :param time_dim: time dimension
    :return:
    """
    if extra_attrs is None:
        extra_attrs = {}
    if time_unit is None:
        time_unit = 'minutes since 1970-01-01'  # units for time in output files
    if time_dim is not None:
        data[time_dim].attrs.pop("units", None)
        data[time_dim].encoding.update(units=time_unit, dtype='float64')
    data.encoding.update(zlib=True, complevel=4)
    data.attrs.update(extra_attrs)
    data.to_netcdf(outpath, unlimited_dims=time_dim)
    my_logger.info(f'Wrote data to {outpath}')


def std_fig_axs(fig_num, reduce_spline: bool = True, regions: bool = False, **kwargs) \
        -> tuple['matplotlib.figure.Figure', 'matplotlib.axes.Axes']:
    mosaic = [['Mornington', 'BLANK', 'Cairns'],
              ['Grafton', 'Brisbane', 'Gladstone'],
              ['Canberra', 'Sydney', 'Newcastle'],
              ['Adelaide', 'Wtakone', 'Melbourne'],
              ]

    if regions:
        mosaic = [row + [rname] for row, rname in zip(mosaic, region_names.keys())]
    args = dict(num=fig_num, clear=True, figsize=(6, 9), layout='constrained',
                empty_sentinel='BLANK', )  # default args,
    args.update(**kwargs)  # update with any passed in args
    fig, axes = plt.subplot_mosaic(mosaic, **args)
    if reduce_spline:  # remove the top and right spines
        for ax in axes.values():
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    return fig, axes


def comp_ratios(gev_parameters: xarray.Dataset, covariance: str | list[str] = 'Tanom') -> xarray.DataArray:
    """
    Compute the ratios of the GEV parameters.
    Args:
        gev_parameters: dataset of GEV parameters
        covariance: covariance (s) to use.
    Returns: xarray data array of the ratios
    """
    if isinstance(covariance, str):
        cov = [covariance]
    else:
        cov = covariance
    da=[]
    for p,c in itertools.product(['location','scale'],cov):
        dp = f'D{p}_{c}'
        ratio = (gev_parameters.sel(parameter=dp)/gev_parameters.sel(parameter=p)).assign_coords(parameter=dp)
        da.append(ratio)
    da=xarray.concat(da,dim='parameter')
    return da

