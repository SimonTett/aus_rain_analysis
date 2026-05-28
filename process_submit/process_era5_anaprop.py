"""
Trial code to mask radar data by ERA-5 reflectivty data.
Will do this for a day at a time. Requires interpolating ERA-5 data to radar data.
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
parser.add_argument('station',choices=ausLib.site_numbers.keys(),help='Radar to use.')
parser.add_argument('time', type=pd.Timestamp, nargs=2, help='Time range to use. Uses year and month and will iterate from start to end in 1 month steps')
parser.add_argument('--end_time', type=pd.Timestamp, help='End time. Times will be a list which will be iterated over')
parser.add_argument('--output_name',  type=pathlib.Path, help='Where to put output.  If no value provided. default will be worked out')

ausLib.add_std_arguments(parser,dask=False) # not much gain from using dask on single pt.
args = parser.parse_args()
my_logger = ausLib.process_std_arguments(args)  # deal with the std arguments (and get the logging)
times = args.time
times = pd.date_range(start=times[0],end=times[1],freq='MS') # make list of times to process. MS means month start.
site_info = ausLib.site_info(args.station).iloc[0]
out_path = args.output_name
if (out_path is not None) and (not args.overwrite) and out_path.exists():
    raise FileExistsError(f"Overwriting existing file {out_path}")

output_data = []
for t in times:
    my_logger.debug(f"Processing data for {t.year}-{t.month:02d}")
    file_list = era5_files(t)
    era5_ds = xarray.open_mfdataset(file_list)
    era5_ts = era5_ds.sel(longitude=site_info.site_lon,latitude=site_info.site_lat,method='nearest')
    if 'valid_time' in era5_ts.coords: # when downloaded have valid_time
        era5_ts = era5_ts.rename(dict(valid_time='time'))
    # select to month of interest
    era5_ts = era5_ts.sel(time=(era5_ts.time.dt.year == t.year) & (era5_ts.time.dt.month == t.month))
    anaprop = (era5_ts.dctb >=0.0) & (era5_ts.dctb < 750.) & (era5_ts.dndzn <= -0.157)
    anaprop = anaprop.rename('Anaprop')
    era5_ts['anaprop']= anaprop.astype('bool') # keep  as bool.

    output_data += [era5_ts]
    start_str = str(era5_ts.time[0].dt.strftime("%Y%m%d").values)
    end_str = str(era5_ts.time[-1].dt.strftime("%Y%m%d").values)
    my_logger.debug(f"Processed data from {start_str} to {end_str}")


output_data = xarray.concat(output_data,dim='time').sortby('time')
if out_path is None:
    start_str = str(output_data.time[0].dt.strftime("%Y%m%d").values)
    end_str = str(output_data.time[-1].dt.strftime("%Y%m%d").values)
    out_path = pathlib.Path(args.station + "_ERA5_anaprop_" + start_str + "_" + end_str + ".nc")
    my_logger.info(f"Writing to {out_path}")
    # done constructing out_file
    if (not args.overwrite) and out_path.exists():
        raise FileExistsError(f"Overwriting existing file {out_path}")

output_data.load() # load the data so we can write it out.
output_data.to_netcdf(out_path)
my_logger.info(f"Saved data to  {out_path}")

    


