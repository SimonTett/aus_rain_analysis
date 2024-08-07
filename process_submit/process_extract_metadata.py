#!/usr/bin/env python
# extract meta data from PPI files.
import sys
import typing

import xradar  # this should happen early so that radar magic can happen!
import xarray
import itertools
import multiprocessing
import argparse
import ausLib
import pathlib
import pandas as pd
import dask
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


# list of variables to drop,
drop_variables = ['time_coverage_start', 'time_coverage_end',
                  'time_reference']  # variables to be dropped from the dataset. This is a global variable. It is updated while running.


def read_ppi(file: pathlib.Path, level1_file: typing.Optional[pathlib.Path] = None) -> xarray.Dataset:
    """
    read meta-data from ppi zip file
    :param file: zip file to be extracted and first file to be read in from
    :param level1_file: level1 file to be read in from -- optional.
    If provided then rapic_VIDRES, rapic_CLEARAIR & rapic_NOISETHRESH will be extracted and added to the meta-data
    :return:  metadata as xarray dataset
    """
    global drop_variables
    vars_to_keep = ['azimuth', 'elevation', 'fixed_angle', 'corrected_reflectivity']
    ds_first = ausLib.read_radar_zipfile(file, first_file=True,
                                         drop_variables=drop_variables)  # just extract metadata from the first file.
    ref = ds_first['corrected_reflectivity']
    calib_off = xarray.DataArray(float(ref.attrs.pop('calibration_offset', np.nan))).assign_attrs(ref.attrs)
    ds_first = ds_first.drop_vars('corrected_reflectivity').assign(calibration_offset=calib_off)
    # check nothing has time in its dims
    has_time = [v for v in ds_first.data_vars if ('time' in ds_first[v].dims
                                                  and v not in vars_to_keep)]
    has_sweep = [v for v in ds_first.data_vars if ('sweep' in ds_first[v].dims and v not in vars_to_keep)]
    if len(has_time) > 0:
        my_logger.warning(f'Variables {has_time} have time in their dimensions.  Adding to drop_var and dropping. ')
        drop_variables += has_time
        ds_first = ds_first.drop_vars(has_time)  # drop any variables that have time in their dims.
    if len(has_sweep) > 0:
        my_logger.warning(f'Variables {has_sweep} have sweep in their dimensions.  Adding to drop_var and dropping. ')
        drop_variables += has_sweep
        ds_first = ds_first.drop_vars(has_sweep)
    # convert coords to data vars
    coords = list(set(ds_first.dims) - {'time', 'sweep'})
    ds_first = ds_first.reset_index(coords).reset_coords(coords)
    # store the unique azimuths and elevations and drop time. Then add a new time var from the min time.
    min_time = ds_first.time.min()
    variables = coords
    variables += vars_to_keep
    # remove corrected_reflectivity from the list of variables to be kept.
    variables.remove('corrected_reflectivity')
    data_vars = dict()
    for v in variables:
        try:
            values = np.unique(ds_first[v])
            if len(values) > 1:
                da = xarray.DataArray(values, coords={f'{v}_index': values})
            else:
                da = xarray.DataArray(values[0])
            data_vars[v] = da
        except KeyError:  # variable not in dataset.
            pass
    #  rename fixed_angle to elevation for consistency,
    if 'fixed_angle' in data_vars.keys():
        data_vars['elevation'] = data_vars.pop('fixed_angle').rename({'fixed_angle_index': 'elevation_index'})
    variables += ['time']
    ds_first = ds_first.drop_vars(variables, errors='ignore').assign(data_vars)
    # convert attributes to data variables
    attrs_want = ['instrument_name', 'time_coverage_resolution']
    attrs_to_vars = {k: ds_first.attrs[k] for k in attrs_want if k in ds_first.attrs.keys()}
    attrs_to_vars['time_coverage_resolution'] = iso_to_timedelta64(ds_first.attrs['time_coverage_resolution'])
    ds_first = ds_first.assign(**attrs_to_vars)
    ds_first['file'] = str(file)
    my_logger.debug(f'Read data from file {file}')
    # now deal with level_1 data.
    if level1_file is not None:
        level1 = ausLib.read_radar_zipfile(level1_file, first_file=True, group='dataset1/data1/how',file_pattern='*.h5')
        for attrib in ['rapic_VIDRES', 'rapic_CLEARAIR', 'rapic_NOISETHRESH']:
            value = level1.attrs.get(attrib, None)
            if value is not None and attrib == 'rapic_CLEARAIR':
                value = (value.lower() == 'true') #
            ds_first[attrib] = value
        ds_first['level1_file'] = str(level1_file)
        my_logger.debug(f'Read data from level1 file {level1_file}')
    ds_first = ds_first.expand_dims(time=[min_time.values])
    return ds_first


if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!

    parser = argparse.ArgumentParser(description='Extract meta-data from PPI files')
    parser.add_argument('site', help='Radar site to process', choices=ausLib.site_numbers.keys())
    parser.add_argument('--years', nargs='+', type=int, help='List of years to process',
                        default=range(1990, 2030))
    parser.add_argument('--outdir', type=pathlib.Path,
                        help='output dir for metadata. If not provided worked out from site')
    ausLib.add_std_arguments(parser,dask=False)  # add on the std args Very I/O intensive. Can't see DASK helping.
    args = parser.parse_args()
    my_logger = ausLib.process_std_arguments(args)  # setup the std stuff

    site = args.site

    extra_attrs = dict(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.utcnow().isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()],
                       site=site)
    for name, value in vars(args).items():
        my_logger.info(f"Arg:{name} =  {value}")

    site_number = f'{ausLib.site_numbers[args.site]:d}'
    indir = pathlib.Path(f'/g/data/rq0/level_1b/{site_number}/ppi')
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

        pattern = f'{site_number}_{year:04d}*_ppi.zip'
        data_dir = indir / f'{year:04d}'
        zip_files = sorted(data_dir.glob(pattern))  #, key=month_sort)
        if len(zip_files) == 0:
            my_logger.info(f'No files found for  pattern {pattern} in {data_dir} {ausLib.memory_use()}')
            continue
        my_logger.info(f'Found {len(zip_files)} files for pattern {pattern} {ausLib.memory_use()} ')
        # these are all daily files. But we want the first one from each month.
        files = []
        ukeys = []
        for k, g in itertools.groupby(zip_files, key=month_sort):
            files.append(list(g)[0])  # just want the first file from each month.
            ukeys.append(k)
        dataset = []
        for f in files:
            my_logger.debug(f'Processing {f}')
            # generate the filename for the level 1 file so we can extract some metadata from that file.
            parts = f.parts
            lev1 = list(parts[0:4]) + ['level_1', 'odim_pvol', parts[5], parts[7], 'vol']
            lev1 += [parts[-1].replace('_ppi.zip', '.pvol.zip')]
            lev1 = pathlib.Path(*lev1)
            if not lev1.exists():
                lev1 = None

            ds = read_ppi(f,level1_file=lev1)
            dataset += [ds]

        dd = xarray.concat(dataset, 'time')
        ausLib.write_out(dd, outfile, extra_attrs=extra_attrs)
