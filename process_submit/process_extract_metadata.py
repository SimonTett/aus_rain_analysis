#!/usr/bin/env python
# extract metadata from PPI files.

import typing

import xradar  # this should happen early so that radar magic can happen!
import xarray
import itertools
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


def extract_convert(ds:xarray.Dataset,
                    attribs:typing.Optional[list[str]]=None)-> dict[str,typing.Union[xarray.DataArray,float]]:
    """
    Extract and converted attributes from xarray dataset
    Args:
        ds: dataset to extract attributes from
        attribs: list of attributes to extract. If none, all attributes are extracted.
        time_attrs: list of attributes which are time attributes.

    Returns: dict of values.

    """
    result = dict()
    if attribs is None:
        attribs = list(ds.attrs.keys())
    names_bool = ['rapic_CLEARAIR'] # attribute names which are logical and if not set should be true
    for name in attribs:
        value = ds.attrs.get(name, None)
        if value is not None and name in names_bool:
            value = (value.lower() == 'true')  #
        result[name] = value
        if isinstance(value, np.ndarray):  # deal with arrays. Want index.
            value = xarray.DataArray(value, coords={f'{name}_index': np.arange(len(value))})
        result[name] = value
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


    result = extract_convert(dt['dataset1/data1/how'].to_dataset(),
                        attribs=['rapic_VIDRES', 'rapic_DBZCOR', 'rapic_CLEARAIR', 'rapic_NOISETHRESH', "rapic_DBZLVL"])

    result.update(extract_convert(dt['/how'].to_dataset(),
                                  attribs=["beamwH", "beamwV","rapic_AZCORR","rapic_ELCORR",
                                           "beamwidth","wavelength","antgainH"]))
    result.update(extract_convert(dt['/where'].to_dataset()))
    result.update(extract_convert(dt['dataset1/how'].to_dataset()))
    result.update(extract_convert(dt['dataset1/where'].to_dataset()))
    result.update(extract_convert(dt['dataset1/what'].to_dataset(), attribs=['startdate', 'starttime']))
    time = pd.to_datetime(result.pop('startdate') + result.pop('starttime')).to_datetime64()

    result['level1_file'] = str(file)
    ds = xarray.Dataset(result).expand_dims(time=[time]) # assign time coord to dataset.
    ds['time'] = ds.time.dt.round("D")
    my_logger.debug(f'Read data from level1 file {file}')

    return ds

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
    vars_to_keep = ['azimuth', 'elevation', 'fixed_angle']
    ds_first = ausLib.read_radar_zipfile(file, first_file=True,
                                         drop_variables=drop_variables,concat_dim='time')  # just extract metadata from the first file.
    ref = ds_first['corrected_reflectivity']
    calib_off = xarray.DataArray(float(ref.attrs.pop('calibration_offset', np.nan))).assign_attrs(ref.attrs)
    ds_first = ds_first.drop_vars('corrected_reflectivity').assign(calibration_offset=calib_off)
    # check nothing has time in its dims
    has_time = [v for v in ds_first.data_vars if ('time' in ds_first[v].dims
                                                  and v not in vars_to_keep)]

    if len(has_time) > 0:
        my_logger.warning(f'Variables {has_time} have time in their dimensions.  Adding to drop_var and dropping. ')
        drop_variables += has_time
        ds_first = ds_first.drop_vars(has_time)  # drop any variables that have time in their dims.
    has_sweep = [v for v in ds_first.data_vars if ('sweep' in ds_first[v].dims and v not in vars_to_keep)]
    if len(has_sweep) > 0:
        my_logger.warning(f'Variables {has_sweep} have sweep in their dimensions.  Adding to drop_var and dropping. ')
        drop_variables += has_sweep
        ds_first = ds_first.drop_vars(has_sweep)
    # convert coords to data vars
    coords = list(set(ds_first.dims) - {'time', 'sweep'})
    ds_first = ds_first.reset_index(coords).reset_coords(coords)
    # store the  azimuths and elevations and drop time. Then add a new time var from the min time.
    min_time = ds_first.time.min()
    variables = coords
    variables += vars_to_keep
    data_vars = dict()
    for v in variables:
        name = v
        #  rename fixed_angle to elevation for consistency,
        if v == 'fixed_angle':
            name = 'elevation'
        try:
            values = np.unique(ds_first[v])
            if len(values) > 1: # store the min and max values, and count for each variable.
                da_min = ds_first[v].min()
                da_max = ds_first[v].max()
                ds_count = xarray.DataArray(len(values))
                ds_first.drop_vars([v])
                data_vars[f'{name}_min'] = da_min
                data_vars[f'{name}_max'] = da_max
                data_vars[f'{name}_count'] = ds_count
            else:
                da = xarray.DataArray(values[0])
                data_vars[v] = da
        except KeyError:  # variable not in dataset.
            pass


    variables += ['time']
    ds_first = ds_first.drop_vars(variables, errors='ignore').assign(data_vars)
    # convert attributes to data variables
    attrs_want = ['instrument_name', 'time_coverage_resolution','time_coverage_duration']
    attrs_to_vars = {k: ds_first.attrs.pop(k)for k in attrs_want}
    for k in ['time_coverage_resolution','time_coverage_duration']: # time things
        v=attrs_to_vars.get(k)
        attrs_to_vars[k] = iso_to_timedelta64(v) if v is not None else None

    ds_first = ds_first.assign(**attrs_to_vars)
    ds_first['ppi_file'] = str(file)
    my_logger.debug(f'Read data from file {file}')

    ds_first = ds_first.expand_dims(time=[min_time.values]) # add on time
    ds_first['time'] = ds_first.time.dt.round("D")
    return ds_first


if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!

    parser = argparse.ArgumentParser(description='Extract meta-data from level 1 and, if they exist, ppi files')
    parser.add_argument('site', help='Radar site to process', choices=ausLib.site_numbers.keys())
    parser.add_argument('--years', nargs='+', type=int, help='List of years to process',
                        default=range(1990, 2030))
    parser.add_argument('--outdir', type=pathlib.Path,
                        help='output dir for metadata. If not provided worked out from site')
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
    indir = ausLib
    indir = pathlib.Path(f'/g/data/rq0/level_1/odim_pvol/{site_number}')
    indir_ppi = pathlib.Path(f'/g/data/rq0/level_1b/{site_number}/ppi')
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

        pattern = f'{site_number}_{year:04d}*.pvol.zip' # level 1 info
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
        # these are all daily files. But we want every 5th day from a month. So group by month and take every 5th.
        files = [list(g)[0] for k, g in itertools.groupby(zip_files, key=month_sort)] # first day of the month
        files = zip_files[::5]  # every 5th day
        # now to build a year's worth of files.
        dataset = []
        for f in zip_files: # all days.
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
        # now processed a year worth of files.
        # concat them all together and write out
        # sometimes haveno data present or it is missing.
        dd = xarray.concat(dataset, 'time',data_vars='all',coords='minimal')
        ausLib.write_out(dd, outfile, extra_attrs=extra_attrs)
