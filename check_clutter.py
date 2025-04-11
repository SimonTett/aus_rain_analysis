# check seasonal maxes against possible clutter values
import re

import numpy as np
import pandas as pd
import xarray
import ausLib

my_logger = ausLib.my_logger
ausLib.init_log(my_logger,level='INFO')
use_cache = False
calib='melbourne'
## compute how much of the max rain rate is clutter
clutter_fraction=[]
site_clutter_times=dict()
for site in ['Sydney','Melbourne','Adelaide','Brisbane','Wtakone']:
    site_dir=ausLib.data_dir/f"site_data/{site}"
    site_dir.mkdir(exist_ok=True,parents=True)
    summary_count_file=site_dir/f'{site}_clutter_count.csv'
    summary_time_file = site_dir/f'{site}_clutter_time.csv'
    my_logger.info(f'Processing {site}')
    radar_seas_mean_file = ausLib.data_dir / f"processed/{site}_rain_{calib}/seas_mean_{site}_rain_{calib}_DJF.nc"
    radar_ds = xarray.open_dataset(radar_seas_mean_file, decode_timedelta=True)
    mx_time = radar_ds.time_max_rain_rate
    mx_time = mx_time.where(mx_time.time.dt.season == 'DJF', drop=True)  # summer
    mx_time = mx_time.sel(time=slice('2010-01-01', '2021-02-28')).load()  # 2010-2020
    if (summary_count_file.exists() and use_cache):
        my_logger.info(f'Using cache for {site}')
        site_count = pd.read_csv(summary_count_file,index_col=0)
        site_clutter = pd.read_csv(summary_time_file,index_col=0)
        site_clutter_times[site] = site_clutter
    else:
        my_logger.info(f'Computing clutter times and counts for  {site}')
        file = ausLib.module_path / f'meta_data/{site.lower()}_clutter_events.csv'
        df = pd.read_csv(file, header=[0],parse_dates=[0,1],dayfirst=True)
        # fix the df
        df = df.apply(pd.to_datetime)

        clutter_dates = df['Start Time'].dt.strftime('%Y-%m-%d')
        df['Date'] = clutter_dates
        clutter_dates = clutter_dates.unique()




        site_clutter=[]
        site_count = []
        for resample in mx_time.resample_prd.values:
            # Timestamps are at the end of the resample period -- see process_reflectivity.py
            end_dt = mx_time.sel(resample_prd=resample).stack(idx=[...]).dropna('idx')
            start_dt = end_dt - pd.Timedelta(str(resample))
            clutter_times = []
            clutter_count=[]
            for indx,series  in df.iterrows():
                my_logger.debug(f'Processing {series["Start Time"]}')
                t_start = series['Start Time']
                in_range = (t_start >= start_dt) & (t_start < end_dt) # values in range
                t_end = series['End Time']
                in_range = in_range | ((t_end >= start_dt) & (t_end < end_dt))
                if in_range.any():
                    clutter_times.append(t_start)
                    clutter_count.append(int(in_range.sum()))

            clutter_times = pd.Series(clutter_times).rename(str(resample))
            clutter_count = pd.Series(clutter_count).rename(str(resample))
            site_clutter.append(clutter_times)
            site_count.append(clutter_count)
        site_clutter = pd.concat(site_clutter, axis=1).astype('<M8[ns]')
        site_count = pd.concat(site_count, axis=1)
        site_count.to_csv(summary_count_file)
        site_clutter.to_csv(summary_time_file)
        site_clutter_times[site] = site_clutter
    #f = lambda x: pd.Timestamp.strftime(x,'%y-%m-%d %H:%M')
    #print(site_clutter.map(f, na_action='ignore')) # extremes where we have clutter

    count_total_max = (~mx_time.isnull()).sum(['x','y','time']).to_dataframe(name='total_max')['total_max']
    clutter_fraction.append((site_count.sum() / count_total_max).rename(site))

clutter_fraction=pd.DataFrame(clutter_fraction)
# Format the DataFrame to show values rounded to two sig figures
def format_sigfig(x):
    return f"{x:.2g}".replace('e', r'\\times 10^{') + '}'

def format_sigfig(x):
    formatted = f"${x:.2g}$"
    formatted = re.sub(r'e([+-])0?(\d+)', r'\\times 10^{\1\2}', formatted)
    return formatted

clutter_fraction=clutter_fraction.loc[:,['30min','1h','2h','4h']]
clutter_fraction.columns = pd.MultiIndex.from_product([['Accumulation Period'],clutter_fraction.columns])
clutter_fraction = clutter_fraction.rename_axis('Site')
styled_clutter_fraction = clutter_fraction.style.\
    format(format_sigfig).\
    set_caption("Clutter fraction")

# Convert the styled DataFrame to a LaTeX table
latex_table = styled_clutter_fraction.to_latex(label='tab:cluter',
                                               caption='Fraction of clutter times corresponding to seasonal max')

# Print the LaTeX table
print(latex_table)
#clutter_fraction.to_csv(ausLib.data_dir/'site_data'/'clutter_fraction.csv')

#print(site,"\n ===================== \n", (site_count.sum() * 100 / count_total_max).round(1))




