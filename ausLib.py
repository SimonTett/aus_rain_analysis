# Library for australian analysis.
import pathlib

import numpy as np
import xarray
import typing
import logging
import pandas as pd


radar_dir = pathlib.Path("/g/data/rq0/level_2/")

my_logger = logging.getLogger(__name__)  # for logging

def gsdp_metadata(files:typing.Iterable[pathlib.Path])  -> pd.DataFrame:
    """
    Read metadata from multiple files into a DataFame. See read_gdsp_metadata for description of columns.
    Args:
        files (): List of files to read in

    Returns: DataFrame of metadata.
    """
    series=[]
    for file in files:
        try:
            s = read_gsdp_metadata(file)
            series.append(s)
        except ValueError: # some problem with data.
            pass
    df = pd.DataFrame(series)
    return df
def read_gsdp_metadata(file:pathlib.Path) -> pd.Series:
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
    result=dict()
    with open(file, 'r') as fh:
        hdr_count=0
        while True:
            key,value = fh.readline().strip().split(":",maxsplit=1)
            # on first line expect key to be 'Station ID'. if not raise an error.
            if (hdr_count == 0) & (key != 'Station ID'):
                raise ValueError(f"File {file} has strange first line {key}:{value}")
            value = value.strip()
            result[key]=value
            hdr_count += 1
            if key == 'Other': # end fo header
                break # all done

    # convert same values to loads
    for key in ['Latitude','Longitude','Resolution','Percent missing data','No data value']:
        result[key]=float(result[key])
    # integer vales
    for key in ['Number of records']:
        result[key]=int(result[key])
    # convert ht.
    result['height']=float(result['Elevation'][0:-1])
    result['height units']=result['Elevation'][-1]
    # rename stuff
    # add in str version of path
    result['file'] = str(file)
    # Store number of header lines -- makes reading in data easier.
    result['header_lines'] = hdr_count
    rename={'New Units':'units',
            'Station ID':'ID',
            'Start datetime':'Start',
            'End datetime':'End',
            'Time Zone':'timezone',
            'New Timestep':'timestep'}
    for old_name, new_name in rename.items():
        result[new_name]=result.pop(old_name)
    # make Start & End datetime
    start = result.pop('Start')
    start = start[0:4]+"-"+start[4:6]+"-"+start[6:8]+"T"+start[8:]
    end = result.pop('End')
    end = end[0:4]+"-"+end[4:6]+"-"+end[6:8]+"T"+end[8:]
    end = pd.to_datetime(end,yearfirst=True,utc=False).tz_localize(result['timezone']).tz_convert('UTC')
    start=pd.to_datetime(start,yearfirst=True,utc=False).tz_localize(result['timezone']).tz_convert('UTC')
    result['End_time']=end
    result['Start_time']=start

    result = pd.Series(result).rename(result['ID'])
    return result

def read_gsdp_data(meta_data:pd.Series) -> pd.Series:
    """
    Read data in given a meta data record
    Args:
        meta_data: meta data record as a pandas series.
    Returns: pandas series with actual information. Missing data is set to null. Will have name = ID

    """
    freq_transform = {'1hr':'1h'} # how we transfer timestep info into pandas freq stings.

    index = pd.date_range(start=meta_data.Start_time,end=meta_data.End_time,freq=freq_transform[meta_data.timestep])
    series = pd.read_csv(meta_data.file,skiprows=meta_data.header_lines-1).set_index(index)
    series = series.iloc[:,0]
    series = series.rename(meta_data.name) # rename
    series = series.astype('float')
    # set all values which are "No data value" to null
    series = series.replace(meta_data['No data value'],np.nan)

    return series


def max_process(ds: xarray.Dataset,
                time_dim: str = 'time',
                minf_mean: float = 0.8,
                minf_max: float = 0.8) -> typing.Optional[xarray.Dataset]:
    """
    Process dataset for max (and mean & time of max).

    Args:
        minf_max: The minimum fraction of samples in ds to compute maximum
        minf_mean: The minimum fraction used to compute a time-mean
        time_dim: the name of the time dimension
        ds: dataset to be processed. Should contain the variables:
            rain -- mean rain
            fraction -- fraction of data present in the mean computation.
            longitude & latitude - longitude and latitude co-ords.
        Note this dataset will be loaded.

    Returns:dataset containing maximum values, time of max values and mean values.
    (or None if no non-missing data).

    """
    ds.load()
    bad = ds.rain.isnull().all(time_dim, keep_attrs=True)  # find out where *all* data null
    if bad.all():
        my_logger.warning(f"All missing at {bad[time_dim].isel[[0, -1]]}")
        return None
    max_rain = ds.rain.max(time_dim, keep_attrs=True, skipna=True)
    max_fraction = (ds.fraction >= minf_mean).mean(time_dim)
    max_rain = max_rain.where(max_fraction >= minf_max).rename('max_rain')
    mean_rain = ds.rain.mean(time_dim, keep_attrs=True, skipna=True)
    mean_rain = mean_rain.where(max_fraction >= minf_max).rename('mean_rain')
    indx = xarray.where(bad, 0.0, ds.rain).argmax(dim=time_dim, keep_attrs=True)
    max_time = ds[time_dim].isel({time_dim: indx}).where(~bad).rename('time_max_rain')  # max times for this season


    base_name = max_rain.attrs['long_name']

    max_fraction.attrs['long_name'] = "Fraction present " + base_name
    max_fraction.attrs['units'] = '1'

    result = xarray.Dataset(
        dict(max_rain=max_rain, time_max_rain=max_time, mean_rain=mean_rain, max_fraction=max_fraction))

    result.attrs.update(ds.attrs)
    return result


def process_gsdp_record(gsdp_record:pd.Series,
                        resample_max:str='QS-DEC'
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
    all = gsdp_record.isnull().resample(resample_max).count()
    fraction = (resamp.count()/all).rename('fraction')
    max = resamp.max().rename("max_rain")
    time_max = resamp.apply(pd.Series.idxmax).rename("time_max_rain")
    # fix the type for time_max
    time_max = time_max.astype(gsdp_record.index.dtype)
    mean = resamp.mean().rename("mean_rain") # need to convert to mm/season?
    total = (mean*all).rename('total_rain')
    df = pd.DataFrame([max,mean,total,fraction,time_max]).T
    for s in ['max_rain','mean_rain','total_rain','fraction']:
        df[s]=df[s].astype('float')
    df['time_max']=df['time_max_rain'].astype(gsdp_record.index.dtype)

    return df

def process_radar(dataSet: xarray.Dataset,
                  mean_resample: str = '1h',
                  min_mean: float = 0.8,
                  max_resample: str = 'QS-DEC',
                  min_max: float = 0.8,
                  time_dim: str = 'time',
                  mask: typing.Optional[xarray.DataArray] = None,
                  radar_range: float = 150e3,
                  ) -> xarray.Dataset:
    """
    Compute max rainfall on mean rainfall from 6 minute radar rainfall data.

      :param dataSet: dataset to be processed. Must contain rainrate, isfile, latitude and longitude
      :param mean_resample: time period to resample data to compute mean
      :param min_mean: min fraction of samples to compute mean.
      :param max_resample : time period to resample meaned data to compute max.
      :param min_max: min fraction of samples to compute max from
      :param time_dim:dimension that is time
      :param mask: Bool mask. Where True when isfile is 1 missing values will be set to 0.0
         If None then will construct mask using radar_range
      :param radar_range: distance from centre to be in radar coverage --
        missing values in this range  will be set to zero if isfile is 1 but only if mask is required.
      :return: max of meaned data with missing data dealt with...


      """
    # TODO add in time-bounds and change time co-ord to be the centre of the interval.
    # Takes about 10 mins to process a year of 1km radar data to monthly data.
    # step 1 -- compute the mean of the radar data,
    my_logger.info("Starting data processing")
    resamp_mean = dataSet.resample({time_dim: mean_resample})
    if mask is None:
        mask = (dataSet.x.astype('float') ** 2 + dataSet.y.astype('float') ** 2) < radar_range ** 2
    # this range is a little large as we have a ring of cells which are always 0.
    rain = []  # list of all rain values
    fraction = []  # fraction of OK values
    for c, ds in resamp_mean:
        coord = {time_dim: c}  # coordinate to set data time
        r = ds.rainrate
        # now to set missing data where data for time is OK and in range of radar to zero.
        r2 = xarray.where(((ds.isfile == 1) & r.isnull() & mask), 0.0, r, keep_attrs=True)
        fract = ds.isfile.mean(time_dim)
        r2 = r2.mean(time_dim, skipna=True, keep_attrs=True)  # compute mean
        r2 = xarray.where(fract >= min_mean, r2, np.nan)
        r2 = r2.assign_coords(coord)  # set the time co-ord
        r2.attrs.update(r.attrs)  # copy attributes back in.
        fract = fract.assign_coords(coord)
        rain.append(r2)
        fraction.append(fract)
        my_logger.debug(f"Processed resample to {mean_resample} data for {c}")

    rain = xarray.concat(rain, dim=time_dim)

    fraction = xarray.concat(fraction, dim=time_dim)
    # set up meta data
    rain.attrs['long_name'] = mean_resample + "mean " + rain.attrs['long_name']
    fraction.attrs['long_name'] = "Fraction present " + rain.attrs['long_name']
    fraction.attrs['units'] = '1'
    fraction.load()  # load the fraction of rain

    ## step 2 -- compute the max using the max resampler.
    resamp_max = xarray.Dataset(dict(rain=rain, fraction=fraction)).resample({time_dim: max_resample})
    result = resamp_max.map(max_process,
                            time_dim=time_dim,
                            minf_mean=min_mean,
                            minf_max=min_max)
    # include lat/long for each var.
    dataSet.latitude.load()
    dataSet.longitude.load()
    for var in ["time_max_rain", "max_rain", "mean_rain"]:
        result[var]=result[var].assign_coords(latitude=dataSet.latitude,
                                              longitude=dataSet.longitude)
    base_name = result.max_rain.attrs['long_name']
    result.max_rain.attrs['long_name'] = max_resample + " max " + base_name
    result.mean_rain.attrs['long_name'] = max_resample + " mean " + base_name
    result.time_max_rain.attrs['long_name'] = max_resample + " max time " + base_name
    # add in the fraction
    fraction2 = fraction.rename({time_dim: time_dim + "_" + mean_resample})  # rename the time dim for fract present
    result['fraction']=fraction2
    result.attrs.update(dataSet.attrs)
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


def seasonal_max(files: typing.Union[typing.List[pathlib.Path], pathlib.Path],
                 outfile: pathlib.Path,
                 resample: str = '1h',
                 variables: typing.Optional[typing.Union[typing.List[str], str]] = None,
                 time_dim: str = 'time',
                 chunks: typing.Optional[typing.Dict[str, int]] = None,
                 overwrite: bool = False) -> xarray.Dataset:
    """
    :param files: List of (or single) pathlib.Path's to be read in and processed
    :param outfile: Output path -- where data will be written to.
    :param resample: resample period
    :param variable:s The variable(s) to be processed.
      If not provided all data variables in the input file(S) will be processed 
   if they have a time coordinate and are not one off: 
           "transverse_mercator", 
           "time_bnds",
           "projection_y_coordinate_bnds", 
           "projection_x_coordinate_bnd. 

    With the exception of time_bdns the first time-slice of these
    variables will be added back in to the processed data.

    :param time_dim: Name of time dimension
    :param chunks: Chunk. See xarray.open_dataset for details.
    :param overwrite: If True overwrite existing data.
    :return: dataset of processed data.

    """

    my_logger.info(f"outfile is: {outfile}")

    if not overwrite and outfile.exists():
        raise FileExistsError(f"{outfile} exists and overwrite is False")
    # convert singleton args into lists.
    if isinstance(variables, str):
        variables = [variables]

    if isinstance(files, pathlib.Path):
        files = [files]

    if chunks is None:
        chunks = {}

    # open up the files
    input_dataset = xarray.open_mfdataset(files, chunks=chunks, concat_dim=time_dim, combine='nested')
    if variables is None:
        variables = list(input_dataset.data_vars.keys())  # list of variables
    # remove bad variables -- must have a time dim.
    variables = [var for var in variables if time_dim in input_dataset[var].coords]
    # and reject the following vars
    vars_noprocess = ["transverse_mercator",
                      "time_bnds",
                      "projection_y_coordinate_bnds",
                      "projection_x_coordinate_bnds"]
    vars_add_back = [var for var in vars_noprocess if var != 'time_bnds']
    variables = [var for var in variables if var not in vars_noprocess]

    my_logger.info(f"Processing the following variables: {' '.join(variables)}")
    # need to take advantage of ancillary variables. Esp fileno = 0 (no data) = 1(data)
    # now read them
    output_datasets = []
    for var in variables:
        my_logger.info(f"Processing {var} with shape {input_dataset[var].shape}")

        ds = resample_max(input_dataset[var], resample=resample, time_dim=time_dim, fillna=0.0)
        ds.load()  # load it
        output_datasets.append(ds)

    output_datasets = xarray.merge(output_datasets)
    my_logger.info("Merged datasets")
    # add in the additional info
    for var in vars_add_back:
        try:
            new_var = input_dataset[var].isel({time_dim: 0}).drop(time_dim).squeeze()
            new_var.load()  # load it
            output_datasets[var] = new_var
            my_logger.info(f"Added {var} to output_dataset")
        except KeyError:  # failed to find what we wanted
            my_logger.info(f"Did not find  {var}")

    my_logger.info("Writing data  out")
    output_datasets.to_netcdf(outfile)  # write out
    my_logger.info(f"Wrote data to {outfile}")

    return output_datasets  # and return.
