#!/usr/bin/env python

""""
Merge and clean metadata
1) Problems to fix -- "no correction should set correction to null"
"""

import pathlib
import xarray
import argparse
import ausLib
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser(description='Merge and clean metadata')
parser.add_argument('input_directory', help='Input directory containing metadata files',type=pathlib.Path)
parser.add_argument('output_path', type=pathlib.Path,
                    help='Output directory for cleaned metadata. If not provided will write into parent of input_directory',nargs="?")
parser.add_argument('--pattern', type=str,default='*_[0-9][0-9][0-9][0-9]_metadata.nc')
ausLib.add_std_arguments(parser,dask=False)
args = parser.parse_args()
my_logger=ausLib.process_std_arguments(args)

if not args.input_directory.is_dir():
    raise FileNotFoundError(f'Input directory {args.input_directory} does not exist or is not a directory')
output_path = args.output_path
if output_path is None:
    output_path = args.input_directory.with_name(args.input_directory.name+'_merge_clean.nc')

if output_path.exists() and (not args.overwrite):
    raise FileExistsError(f"output: {output_path} already exists. Run with --overwrite to overwrite it")

files = sorted(args.input_directory.glob(args.pattern))
my_logger.info(f'Found {len(files)} files for pattern {args.pattern}')
dataset = [xarray.open_dataset(f).assign(metadata_filename=str(f)) for f in files] # open up all the files.

for ds in dataset:
    # fix the calibration_offset -- setting it to nan when have no correction (instead of 0)
    try:
        cal = ds['calibration_offset']
        if cal.attrs.get('calibration_comment','') in ['time period not found in cal file']:
            OK = (cal != 0.0)
            ds['calibration_offset'] = cal.where(OK)
            my_logger.info(f'Set {int((~OK).count())} calibration_offset values to nan in {cal.time.values[0]}')
    except KeyError:
        my_logger.warning(f'No calibration_offset variable in {ds.time.values[0]}')
    # Check all variables are float, int,bool, or timedelta64[ns] EXCEPT for polmode, ppi_file & level1_file
    for variab in ds.data_vars:
        dtype = ds[variab].dtype
        if variab in ['polmode','ppi_file','level1_file', "instrument_name"]: # strings.
            if dtype.kind in ['U','S']:
                continue
            # now have some problem as these should be strings
            raise ValueError("Variable {variab} @ {ds.time.values[0]} has dtype {dtype}. Should be string")
        if ds[variab].dtype not in ['float32','float64','int32','int64','timedelta64[ns]',"bool"]:
            my_logger.warning(f'Variable {variab} @ {ds.time.values[0]} has dtype {dtype}. Converting to float')
            ds[variab] =  xarray.apply_ufunc(
                pd.to_numeric,
            ds[variab],
                kwargs={"errors": "coerce"},   # failed parses -> NaN
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
        )
    # fix the occasional glitch in rapic_DBZLVL by forward filling up to 2 slots (10 days).
    variab = 'rapic_DBZLVL'
    filled = ds[variab].ffill('time',limit=6)  # back fill up to 6 . Data being 5-day


    # if we still have 0 rapid_DBZLVL then find cases with zero and fill them with median values.
    count_levels = filled.notnull().sum(variab+"_index")
    if count_levels.min() == 0:
        my_logger.warning(f'Found {int((count_levels==0).sum())} cases with 0 levels in {variab} @ {ds.time.values[0]}. Filling with median value')
        med = filled.median('time')
        filled = filled.where(count_levels > 0, med)

    ds[variab] = filled

## before merge we need to prep the data by generating "empty" variables.
# dict of variables and their dimensions
fill_value = -32768
info=dict()
info["dims"] = dict()
info['types'] = dict()
for ds in dataset:
    for var in ds.data_vars:
        info['dims'][var]=ds[var].dims
        info['types'][var] = ds[var].dtype # will just use the last one.

for ds in dataset:
    for var in info['types'].keys():
        # Create with fill value for int vars
        if var not in ds.data_vars and info['types'][var].kind in ['i','u']:
            dtype = info['types'][var]
            dims = info['dims'][var]

            # Get the size for each dimension from existing variables in ds
            shape = tuple(ds.sizes[d] for d in dims)

            # Create the array with the appropriate fill value
            data = np.full(shape, fill_value, dtype=dtype)


            # Create DataArray preserving coords
            coords = {d: ds.coords[d] for d in dims if d in ds.coords}
            ds[var] = xarray.DataArray(data, dims=dims, coords=coords)
            ds[var].attrs['_FillValue'] = fill_value
        elif var in ds.data_vars and info['types'][var].kind in ['i','u']:
            ds[var] = ds[var].fillna(fill_value)


##
merged = xarray.concat(dataset, dim='time',coords='minimal').sortby('time')
# now have merged data lets do some fixes. Add in the number of levels and deal with cases with 0 which arise
# from occasional glitch.
# 1) for some string vars get nan.  Convert then to "NAN" and back to string
merged = merged.map(lambda x: x.fillna("NAN") if x.dtype.kind in ['U','S','O'] else x)
merged = merged.map(lambda x: x.astype("U") if x.dtype.kind == "O" else x)
# 2) Forward fill calibration_offset to fill in gaps.
# then unlimited forward fill = "persist"
merged['calibration_offset'] = merged['calibration_offset'].ffill('time', limit=None)
extra_attrs = dict(program_name=str(pathlib.Path(__file__).name),
                   utc_time=pd.Timestamp.utcnow().isoformat(),
                   program_args=[f'{k}: {v}' for k, v in vars(args).items()])
ausLib.write_out(merged,output_path,extra_attrs=extra_attrs)








