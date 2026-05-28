#!/usr/bin/env python
"""

Generate boolian mask for potential anaprop from ERA5 data.
Relevant era-5 variables are read in extracted to station, anaprop
condition applied and then all written out.


Mask condition (at site) is:
(dctb >= 0.0 ) & (dctb < 750.) & (dndzn < -0.157)
>=0 means not missing. Want close enough to ground and want ducting

"""
import pathlib
import typing

import xarray
import pandas as pd
import argparse


import ausLib


def era5_files(time:pd.Timestamp) -> pathlib.Path|list[pathlib.Path]:
    """
    Return era5 file or list of files that match the desired time.

    Args:
        time: time wanted. Used to select files. All data for year will be extracted
        Exact behaviour depends on where is being run

    """

    if ausLib.platform == 'geos':
        gpattern = f"era5_refractivity_{time.year}_*.nc"
        era_file_list = list(ausLib.era5_dir.glob(gpattern))
        if not era_file_list:
            raise ValueError(f" {ausLib.era5_dir}.glob({gpattern}) failed to find any files.")
        my_logger.debug(f"Found {len(era_file_list)} files")

    elif ausLib.platform == 'gadi':
        # find list of files. gadi keeps them for each year/month and grouped by var.
        vars=['tplt','tplb','dctb','dndza','dndzn'] # all vars we need
        era_file_list = []
        for var in vars:
            dir = ausLib.era5_dir/f'single-levels/reanalysis/{var}/{time.year}'
            files = list(dir.glob(f'{var}_era5_oper_sfc_{time.year:04d}{time.month:02d}01*.nc'))
            era_file_list += files
            my_logger.debug(f"var: {var} found nfiles:{len(files)}")
    else:
        raise ValueError(f"Don't know how to find era5 files on platform {ausLib.platform}")


    return era_file_list


parser = argparse.ArgumentParser(description='Compute, form ERRA-5 data, when potn anaprop might occurr')
parser.add_argument('site',choices=ausLib.site_numbers.keys(),help='Radar to use.')
parser.add_argument('time_range', type=pd.Timestamp, nargs=2, help='Time range to use. Uses year and month and will iterate from start to end in 1 month steps')
parser.add_argument('--end_time', type=pd.Timestamp, help='End time. Times will be a list which will be iterated over')
parser.add_argument('--output_file',  type=str,
                    help='File name for output.  If no value provided. default will be worked out')
parser.add_argument('--output_dir',  type=pathlib.Path, help='Directory to write data',
                    default=pathlib.Path.cwd())

ausLib.add_std_arguments(parser,dask=False) # not much gain from using dask on single pt.
args = parser.parse_args()

extra_attrs = dict(program_name=str(pathlib.Path(__file__).name),
                   utc_time=pd.Timestamp.now('UTC').isoformat(),
                   program_args=[f'{k}: {v}' for k, v in vars(args).items()],
                   site=args.site)
my_logger = ausLib.process_std_arguments(args)  # deal with the std arguments (and get the logging)
times = args.time_range
times = pd.date_range(start=times[0],end=times[1],freq='MS') # make list of times to process. MS means month start.
site_info = ausLib.site_info(args.site).iloc[0]
site_lon=site_info.site_lon
site_lat=site_info.site_lat
out_path = args.output_file
if out_path is not None:
    out_path = args.output_dir/out_path
    if (not args.overwrite) and out_path.exists():
        raise FileExistsError(f"Overwriting existing file {out_path}")

output_data = []
for t in times:
    my_logger.debug(f"Processing data for {t.year}-{t.month:02d}")
    file_list = era5_files(t)
    era5_ds = xarray.open_mfdataset(file_list)
    era5_ts = era5_ds.sel(longitude=site_lon,latitude=site_lat,method='nearest')
    if 'valid_time' in era5_ts.coords: # when downloaded have valid_time
        era5_ts = era5_ts.rename(dict(valid_time='time'))
    # select to month of interest
    era5_ts = era5_ts.sel(time=(era5_ts.time.dt.year == t.year) & (era5_ts.time.dt.month == t.month))
    anaprop = (era5_ts.dctb >=0.0) & (era5_ts.dctb < 750.) & (era5_ts.dndzn <= -0.157)
    anaprop = anaprop.rename('Anaprop')
    anaprop.assign_attrs(condition='Cases where 0.0 <= dctb < 750 and dndzn < -0.157')
    era5_ts['anaprop']= anaprop.astype('bool') # keep  as bool.

    output_data += [era5_ts]
    start_str = str(era5_ts.time[0].dt.strftime("%Y%m%d").values)
    end_str = str(era5_ts.time[-1].dt.strftime("%Y%m%d").values)
    my_logger.debug(f"Processed data from {start_str} to {end_str}")


output_data = xarray.concat(output_data,dim='time').sortby('time')
if out_path is None:
    start_str = str(output_data.time[0].dt.strftime("%Y%m%d").values)
    end_str = str(output_data.time[-1].dt.strftime("%Y%m%d").values)
    out_path =args.site + "_ERA5_anaprop_" + start_str + "_" + end_str + ".nc"
    out_path = args.output_dir/out_path
    my_logger.info(f"Constructed out_path: {out_path}")
    # done constructing out_file
    if (not args.overwrite) and out_path.exists():
        raise FileExistsError(f"Overwriting existing file {out_path}")

ausLib.write_out(output_data,out_path,extra_attrs=extra_attrs)


    


