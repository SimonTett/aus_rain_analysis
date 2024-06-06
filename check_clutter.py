# check seasonal maxes against possible clutter values


import numpy as np
import pandas as pd
import xarray
import ausLib
site = 'Melbourne'
radar_seas_mean_file = ausLib.data_dir/f"summary_reflectivity/processed/{site}_hist_gndrefl_DJF.nc"
radar_ds = xarray.open_dataset(radar_seas_mean_file)

# monthly data
in_radar = sorted((ausLib.data_dir/f"summary_reflectivity/{site}").glob("hist_gndrefl*.nc"))
radar_ds = xarray.open_mfdataset(in_radar)
mx_time = radar_ds.time_max_reflectivity
file = ausLib.module_path / 'meta_data/melbourne_possible_clutter.csv'
df = pd.read_csv(file, header=[0],parse_dates=[0,1],dayfirst=True)
clutter_dates = df['Start Time'].dt.strftime('%Y-%m-%d')
df['Date'] = clutter_dates
clutter_dates = clutter_dates.unique()

mx_time = mx_time.where(mx_time.time.dt.season=='DJF',drop=True) # summer
mx_time = mx_time.sel(time=slice('2010-01-01','2021-02-28')).load() # 2010-2020

##
df_clutter=[]
df_count = []
for resample in mx_time.resample_prd.values:
    start_dt = mx_time.sel(resample_prd=resample).stack(idx=[...]).dropna('idx')
    end_dt = start_dt + pd.Timedelta(str(resample))
    clutter_times = []
    clutter_count=[]
    for indx,series  in df.iterrows():
        t_start = series['Start Time']
        in_range = (t_start >= start_dt) & (t_start < end_dt) # values in range
        t_end = series['End Time']
        in_range = in_range | ((t_end >= start_dt) & (t_end < end_dt))
        if in_range.any():
            clutter_times.append(t_start)
            clutter_count.append(int(in_range.sum()))

    clutter_times = pd.Series(clutter_times).rename(str(resample))
    clutter_count = pd.Series(clutter_count).rename(str(resample))
    df_clutter.append(clutter_times)
    df_count.append(clutter_count)
df_clutter = pd.concat(df_clutter,axis=1).astype('<M8[ns]')
df_count = pd.concat(df_count,axis=1)
f = lambda x: pd.Timestamp.strftime(x,'%y-%m-%d %H:%M')
print(df_clutter.map(f,na_action='ignore')) # extremes where we have clutter
print(df_count)




## Summary of findings.
# for the seasonal DFJ max have no 30min, 1h & 2h max contaminated by anomalous prop.
# The 4h and 8h max have 1 event impacted by anom prop/clutter and only 1 cell for each on
# 2013-12-05 17:24
# a bit more contamination in the monthly max. Here have upto 4 max impacted with at most
# three cells at the same time.
# overall don't think I need to worry.
