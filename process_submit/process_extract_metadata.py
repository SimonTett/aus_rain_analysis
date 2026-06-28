#!/usr/bin/env python
# extract metadata from PPI files.

import typing

#import xradar  # this should happen early so that radar magic can happen!
import xarray
import multiprocessing
import argparse
import ausLib
import pathlib
import pandas as pd
import numpy as np
import re

my_logger = ausLib.my_logger


def month_sort(path: pathlib.Path) -> str:
    """
    Sort fn to give yyyy-mm order.
    Args:
        path (): pathlib.path

    Returns: yyyy-mm


    """
    yyyymm = path.name.split('_')[1][0:6]
    return yyyymm

def iso_to_timedelta64(iso_string: str) -> np.timedelta64:
    # Parse the ISO string
    match = re.match(r'P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?', iso_string)
    days, hours, minutes, seconds = match.groups()

    # Convert each time component to a timedelta64 and add them together
    timedelta = np.timedelta64(0, 's')
    if days is not None:
        timedelta += np.timedelta64(int(days), 'D')
    if hours is not None:
        timedelta += np.timedelta64(int(hours), 'h')
    if minutes is not None:
        timedelta += np.timedelta64(int(minutes), 'm')
    if seconds is not None:
        timedelta += np.timedelta64(int(float(seconds)), 's')

    timedelta += np.timedelta64(0, 'ns')  # convert to nano-secs to supress annoying warning.

    return timedelta

def extract_convert(attributes:dict,
                    keys:typing.Optional[list[str]]=None,
                    time_range_attribs:typing.Optional[list[str]]=None,
                    bool_attribs:typing.Optional[list[str]] =None,
                    index_attribs:typing.Optional[list[str] ]= None)-> dict[str,typing.Union[xarray.DataArray,float]]:
    """
    Extract and convert attributes from xarray dataset
    Args:
        attributes: dict to extract attributes from
        keys: list of attributes to extract. If none, all attributes are extracted.
        time_range_attribs: list of attributes to be converted to time range.
        bool_attribs: list of attributes to be converted to bool.
        index_attribs: list of attributes to be kept as 2d arrays

    Returns: dict of values.

    """
    if bool_attribs is None:
        bool_attribs = []
    if index_attribs is None:
        index_attribs = []
    if time_range_attribs is None:
        time_range_attribs = []
    result = dict()
    if keys is None:
        keys = list(attributes.keys())
    for name in keys:
        value = attributes.get(name, None)

        if name in bool_attribs:
            if value is None:
                value = False
            else:
                value = value.lower() == 'true'
            result[name] = value
        elif (name in time_range_attribs) and (value is None or isinstance(value,str)):
            if value is None:
                result[name] = np.timedelta64(0, 's')
            else:
                result[name] = iso_to_timedelta64(value)
        elif isinstance(value, (np.ndarray,list)):  # deal with arrays. May want index.
            if len(value) > 1: # got a numpy array or list.
                if isinstance(value,list):
                    value = [float(v) for v in value] # convert potn strings to floats.
                count = len(value)
                mx = np.max(value)
                mn = np.min(value)
                result[name+"_count"] = count
                result[name+"_min"] = mn
                result[name+"_max"] = mx
                if name in index_attribs:
                    result[name] = xarray.DataArray(value,
                                                    coords={f'{name}_index': np.arange(count)})
            else:
                result[name] = float(value[0])
        else:
            result[name] = value
    # end loop over attribs
    return result

def read_level1_meta(file: pathlib.Path) -> xarray.Dataset:
    """
    read metadata from level 1 pvol.zip file and convert to a xarray dataset.
    :param file: path to the file. Should be a level1 pvol.zip file.
    :return: xarray dataset with metadata.
    """
    # Want all data in /how, /where, dataset1/how & dataset1/data1/how
    if file.suffixes != ['.pvol','.zip']:
        raise ValueError(f"File {file} is not a level1 pvol zip file")
    dt = ausLib.read_radar_zipfile(file, first_file=True, datatree=True,file_pattern='*.h5')

    bool_attribs = ['rapic_CLEARAIR']
    result = extract_convert(dt['dataset1/data1/how'].attrs,
                        keys=['rapic_VIDRES', 'rapic_DBZCOR', 'rapic_CLEARAIR', 'rapic_NOISETHRESH', "rapic_DBZLVL"],
                             index_attribs=['rapic_DBZLVL'],bool_attribs=bool_attribs)

    result.update(extract_convert(dt['/how'].attrs,
                                  keys=["beamwH", "beamwV","rapic_AZCORR","rapic_ELCORR",
                                           "beamwidth","wavelength","antgainH"]))
    result.update(extract_convert(dt['/where'].attrs))
    result.update(extract_convert(dt['dataset1/how'].attrs))
    result.update(extract_convert(dt['dataset1/where'].attrs))
    result.update(extract_convert(dt['dataset1/what'].attrs,
                                  keys=['startdate', 'starttime']))
    time = pd.to_datetime(result.pop('startdate') + result.pop('starttime')).to_datetime64()

    result['level1_file'] = str(file)
    ds = xarray.Dataset(result).expand_dims(time=[time]) # assign time coord to dataset.
    ds['time'] = ds.time.dt.round("D")
    my_logger.debug(f'Read data from level1 file {file}')

    return ds

# list of variables to drop,
drop_variables = ['time_coverage_start', 'time_coverage_end',
                  'time_reference']  # variables to be dropped from the dataset. This is a global variable. It is updated while running.
def read_ppi_meta(file: pathlib.Path) -> xarray.Dataset:
    """
    read metadata from ppi zip file
    :param file:  path to file to be extracted and the first file to be read in from
    :return:  metadata as xarray dataset
    """
    # check that the file is a ppi file.
    if not (file.suffix == '.zip' and file.stem.endswith('_ppi')):
        raise ValueError(f"File {file} is not a ppi zip file")
    global drop_variables # variables dropped from the dataset.
    vars_to_keep = ['azimuth', 'fixed_angle','elevation']
    ds_first = ausLib.read_radar_zipfile(file, first_file=True,
                                         drop_variables=drop_variables,concat_dim='time')  # just extract metadata from the first file.
    # do want to read corrected_reflectivity as want its attributes, so don't drop it.
    ref_var = 'corrected_reflectivity'
    ref = ds_first[ref_var]
    ds_first = ds_first.drop_vars([ref_var])# drop here BEFORE drop_dims check below means corrected_reflectivity is not dropped.
    # We just want its attributes, though.
    calib_off = xarray.DataArray(float(ref.attrs.pop('calibration_offset', np.nan))).assign_attrs(ref.attrs)
    # Find vars that have time or sweep in their dims and drop them in future reads.
    drop_dims = {'time', 'sweep'}  # variable get dropped if they have these dims and are not in vars_to_keep
    drop_vars = [v for v in ds_first.data_vars if
                 (drop_dims.intersection(set(ds_first[v].dims)) and v not in vars_to_keep)]

    if len(drop_vars) > 0:
        my_logger.warning(
            f'Variables {drop_vars} have one or more of {drop_dims} in their dimensions.  Adding to drop_variables  ')
        drop_variables += drop_vars
    # use extract_convert to flatten vars_to_keep into a dict.
    values_dict = {v:np.atleast_1d(ds_first[v].isel(time=0).values) for v in vars_to_keep}
    values_dict['elevation'] = values_dict.pop('fixed_angle') # rename fixed_angle to elevation for consistency overwriting exising elevation
    result = extract_convert(values_dict,index_attribs=['elevation'],)
    # add in relevant attributes
    attrs_to_vars = extract_convert(ds_first.attrs,
                                    keys=['instrument_name', 'time_coverage_resolution', 'time_coverage_duration','instrument_id',
                                          'origin_altitude','origin_latitude','origin_longitude'],
                                    time_range_attribs=['time_coverage_resolution', 'time_coverage_duration'],
                                    )
    result.update(attrs_to_vars) # add to result
    # extract the calibration_offset

    result['calibration_offset'] = calib_off
    # add a new time var from the min time.
    min_time = ds_first.time.min()
    result['ppi_file'] = str(file)

    result = xarray.Dataset(result).expand_dims(time=[min_time.dt.round("D").values])#

    my_logger.debug(f'Read data from file {file}')

    return result


if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!

    parser = argparse.ArgumentParser(description='Extract meta-data from level 1 and, if they exist, ppi files')
    parser.add_argument('site', help='Radar site to process', choices=ausLib.site_numbers.keys())
    parser.add_argument('--years', nargs='+', type=int, help='List of years to process',
                        default=range(1990, 2030))
    parser.add_argument('--outdir', type=pathlib.Path,
                        help='output dir for metadata. If not provided worked out from site')
    parser.add_argument('--glob',  help='glob pattern for level1 data.',default='*.pvol.zip')
    ausLib.add_std_arguments(parser,dask=False)  # add on the std args Very I/O intensive. Can't see DASK helping.
    args = parser.parse_args()
    my_logger = ausLib.process_std_arguments(args)  # set up the std stuff

    site = args.site

    extra_attrs = dict(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.utcnow().isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()],
                       site=site)
    for name, value in vars(args).items():
        my_logger.info(f"Arg:{name} =  {value}")

    site_number = f'{ausLib.site_numbers[args.site]:d}'
    indir = ausLib.level1_dir / f'{site_number}'
    indir_ppi = ausLib.level1b_dir / f'{site_number}/ppi'
    my_logger.info(f'Input directory is {indir}')
    if not indir.exists():
        raise FileNotFoundError(f'Input directory {indir} does not exist')
    outdir = args.outdir
    if outdir is None:
        outdir = ausLib.data_dir / f'site_data/{site}_metadata'
    outdir.mkdir(parents=True, exist_ok=True)
    my_logger.info(f'Output dir is {outdir}')

    for year in args.years:
        outfile = outdir / f'{site}_{year:04d}_metadata.nc'
        my_logger.info(f'Processing year {year}')
        if outfile.exists() and not args.overwrite:
            my_logger.warning(f'Output file {outfile} exists and will not be overwritten. Set --overwrite. Skipping.')
            continue

        pattern = f'{site_number}_{year:04d}{args.glob}' # level 1 info
        data_dir = indir / f'{year:04d}/vol'
        data_dir_ppi = indir_ppi / f'{year:04d}'
        if not data_dir.exists():
            my_logger.warning(f'Directory {data_dir} does not exist. Skipping')
            continue
        zip_files = sorted(data_dir.glob(pattern))  #, key=month_sort)
        if len(zip_files) == 0:
            my_logger.info(f'No files found for  pattern {pattern} in {data_dir} {ausLib.memory_use()}')
            continue
        my_logger.info(f'Found {len(zip_files)} files for pattern {pattern} {ausLib.memory_use()} ')
        # these are all daily files. But we want every 5th day
        # files = [list(g)[0] for k, g in itertools.groupby(zip_files, key=month_sort)] # first day of the month. Legacy
        # now to build a year's worth of files.
        dataset = []
        for f in zip_files[::5]: # every 5th day
            my_logger.debug(f'Processing {f}')
            ds = read_level1_meta(f)  # read the level 1 file to get the rapic and other useful attributes
            # work out the ppi file.
            ppi_file =indir_ppi/f'{year:04d}'/f.name.replace('.pvol.zip','_ppi.zip')
            if ppi_file.exists():  # merge in the ppi file if it exists.
                ds2 = read_ppi_meta(ppi_file)
                ds = xarray.merge([ds,ds2]) # merge the datasets.
            else:
                my_logger.warning(f'No ppi file {ppi_file} found for {f} ')

            dataset += [ds]
        # now processed a years' worth of files.
        # concat them all together and write out
        # sometimes have no data present or it is missing.
        dd = xarray.concat(dataset, 'time',data_vars='all',coords='minimal')
        ausLib.write_out(dd, outfile, extra_attrs=extra_attrs)
