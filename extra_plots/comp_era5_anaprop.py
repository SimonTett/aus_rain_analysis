"""
Trial code to mask radar data by ERA-5 reflectivty data.
Will do this for a day at a time. Requires interpolating ERA-5 data to radar data.
Mask condition (at site) is:
(dctb >= 0.0 ) & (dctb < 750.) & (dndzn < -0.157)
>=0 means not missing. Want close enough to ground and want ducting
"""
import pathlib
import typing

import cartopy.crs as ccrs
import xarray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

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
        era_file_list = ausLib.era5_dir.glob(gpattern)
        if not era_file_list:
            raise ValueError(f" {ausLib.era5_dir}.glob({gpattern}) failed to find any files.")

    elif ausLib.platform == 'gadi':
        # find list of files. gadi keeps them for each year/month and grouped by var.
        vars=['tplt','tplb','dctb','dndza','dndzn'] # all vars we need
        era_file_list = []
        for var in vars:
            dir = ausLib.era5_dir/f'single-levels/reanalysis/{var}/{time.year}'
            files = list(dir.glob(f'{var}_era5_oper_sfc_{time.year:04d}{time.month:02d}01*.nc'))
            era_file_list += files
            logging.info(f"var: {var} files:{files}")


    return era_file_list


sent_output=object()
parser = argparse.ArgumentParser(description='Compute when potn anaprop might occurr')
parser.add_argument('station',choices=ausLib.site_numbers.keys(),help='Radar to use.')
parser.add_argument('time', type=pd.Timestamp, help='Time to use. Uses year and month')
parser.add_argument('--end_time', type=pd.Timestamp, help='End time. Times will be a list which will be iterated over')
parser.add_argument('--output', nargs='?', type=pathlib.Path, help='Where to put output.  If no value provide. default will be worked out',const=sent_output)


args = parser.parse_args()
time: pd.Timestamp = args.time
if args.end_time:
    time = pd.date_range(time,end=args.end_time,freq='MS')
else:
    time=[time]
site_info = ausLib.site_info(args.station).iloc[0]


output_data = []
for t in time:
    file_list = era5_files(t)
    era5_ds = xarray.open_mfdataset(file_list)
    era5_ts = era5_ds.sel(longitude=site_info.site_lon,latitude=site_info.site_lat,method='nearest')
    # now to load it. Which will take a while...
    era5_ts.load()
    anaprop = (era5_ts.dctb >=0.0) & (era5_ts.dctb < 750.) & (era5_ts.dndzn <= -0.157)
    anaprop = anaprop.rename('Anaprop')
    era5_ts['anaprop']= anaprop.astype('bool') # keep  as bool.
    if 'valid_time' in era5_ts.coords: # when downloaded have valid_time
        era5_ts = era5_ts.rename(dict(valid_time='time')) 
    output_data += [era5_ts]
    print(f"{args.station} {t} {int(anaprop.sum())/float(anaprop.count()):0.3f}")

out_file = args.output


if out_file:
    output_data = xarray.concat(output_data,dim='time')
    output_data = output_data.sortby('time')
    if out_file is sent_output: # need to set output
        start_str = str(output_data.time[0].dt.strftime("%Y%m%d").values)
        end_str = str(output_data.time[0].dt.strftime("%Y%m%d").values)

        out_file = args.station+"_ERA5_anaprop_"+start_str+"_"+end_str+".nc"
    # done constructing out_file
    output_data.to_netcdf(out_file)
    print(f"Saved data to  {out_file}")

    


