# Library for australian analysis.
import pathlib

import numpy as np
import xarray
import typing
import logging
import resource

radar_dir = pathlib.Path("/g/data/rq0/level_2/")

my_logger = logging.getLogger(__name__)  # for logging


def process_radar(dataSet: xarray.Dataset,
                  mean_resample: str = '1h',
                  min_mean: float = 0.8,
                  max_resample: str = 'QS-DEC',
                  min_max: float = 0.8,
                  time_dim: str = 'time',
                  radar_range: float = 150e3,
                  ) -> xarray.Dataset:
    """
    Compute max rainfall on mean rainfall from 6 minute radar rainfall data.

      :param dataSet: dataset to be processed. Must contain rainrate & isfile
      :param mean_resample: time period to resample data to compute mean
      :param min_mean: min fraction of samples to compute mean.
      :param max_resample : time period to resample meaned data to compute max.
      :param min_max: min fraction of samples to compute max from
      :param time_dim:dimension that is time
      :param radar_range: distance from centre to be in radar coverage --
        missing values in this range  will be set to zero if isfile is 1.
      :return: max of meaned data with missing data dealt with...


      """
    # TODO add in time-bounds and change time co-ord to be the centre of the interval.
    # Takes about 10 mins to process a year of 1km radar data to monthly data.
    # step 1 -- compute the mean of the radar data,
    my_logger.info("Starting data processing")
    resamp_mean = dataSet.resample({time_dim: mean_resample})
    in_range = (dataSet.x.astype('float') ** 2 + dataSet.y.astype('float') ** 2) < radar_range ** 2
    # this range is a little large as we have a ring of cells which are always 0.
    rain = []  # list of all rain values
    fraction = []  # fraction of OK values
    for c, ds in resamp_mean:
        coord = {time_dim: c}  # coordinate to set data time
        r = ds.rainrate
        # now to set missing data where data for time is OK and in range of radar to zero.
        r2 = xarray.where(((ds.isfile == 1) & r.isnull() & in_range), 0.0, r, keep_attrs=True)
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
    max_rain = []
    max_fraction = []
    max_time = []
    mean_rain = []
    for c, ds in resamp_max:
        ds.load()
        bad = ds.rain.isnull().all(time_dim, keep_attrs=True)  # find out where *all* data null
        if bad.all():
            my_logger.warning(f"All missing at {c}")
            continue  # skip further processing as all missing.
        coord = {time_dim: c}
        r = ds.rain.max(time_dim, keep_attrs=True, skipna=True)
        f = (ds.fraction >= min_mean).mean(time_dim).assign_coords(coord)
        r = r.where(f >= min_max).rename('max_rain').assign_coords(coord)
        mn_r = ds.rain.mean(time_dim, keep_attrs=True, skipna=True)
        mn_r = mn_r.where(f >= min_max).rename('mean_rain').assign_coords(coord)
        indx = xarray.where(bad, 0.0, ds.rain).argmax(dim=time_dim, keep_attrs=True)
        mtime = ds[time_dim].isel({time_dim: indx}).where(~bad).rename('time_max_rain').assign_coords(
            coord)  # max times for this season
        max_rain.append(r)
        max_fraction.append(f)
        max_time.append(mtime)
        mean_rain.append(mn_r)
        my_logger.info(f"Max processing {max_resample} done for {c}")

    max_rain = xarray.concat(max_rain, dim=time_dim)
    max_time = xarray.concat(max_time, dim=time_dim)
    max_fraction = xarray.concat(max_fraction, dim=time_dim)
    mean_rain = xarray.concat(mean_rain, dim=time_dim)
    # fix meta data
    base_name = max_rain.attrs['long_name']
    max_rain.attrs['long_name'] = max_resample + " max " + base_name
    mean_rain.attrs['long_name'] = max_resample + " mean " + base_name
    max_time.attrs['long_name'] = max_resample + " max time " + base_name
    max_fraction.attrs['long_name'] = "Fraction present " + base_name
    max_fraction.attrs['units'] = '1'
    fraction2 = fraction.rename({time_dim: time_dim + "_" + mean_resample})  # rename the time dim for fract present
    result = xarray.Dataset(
        dict(max_rain=max_rain, max_rain_time=max_time, mean_rain=mean_rain, max_fraction=max_fraction,
             fraction=fraction2))
    result.attrs.update(dataSet.attrs)
    my_logger.info(f"Completed processing of {len(result.fraction)} hours")
    return result


def resample_max(data_array: xarray.DataArray,
                 resample: str = '1h',
                 time_dim: str = 'time',
                 fillna: typing.Optional[float] = None) -> xarray.Dataset:
    """

    :param data_array: data array to be processed.
    :param resample: resample value 
    :param time_dim: name of time dimension
    :param fixNan -- if True set Nan to zero.
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
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        my_logger.debug(f"Loaded data for coord {c}. Memory = {mem} GBytes")
        indx = da.argmax(dim=time_dim, skipna=True, keep_attrs=True)  # index of maxes
        coord = {time_dim: c}  # coordinate to set data time
        mtime = da[time_dim].isel({time_dim: indx}).where(~bad).rename('time_max').assign_coords(
            coord).load()  # max times for this season
        mvalue = da.isel({time_dim: indx}).where(~bad).assign_coords(coord).load()
        # max value for this season
        # then store things.
        mx_time.append(mtime)
        mx_value.append(mvalue)
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # in Gbytes
        my_logger.debug(f"Computed mx value and time of max at time {c}. Mem = {mem} Gbytes")
    mx_time = xarray.concat(mx_time, dim=time_dim).sortby(time_dim)  # concat everything
    mx_value = xarray.concat(mx_value, dim=time_dim).sortby(time_dim)
    my_logger.debug(f"Concatted. Mem = {mem}")
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
