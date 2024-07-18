# find the radar stations that have long records. Print out relevant meta-data
import pathlib

import pandas as pd
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns

import ausLib


def print_df(df):
    """
    Print out a data frame!
    :param df:
    :return:
    """
    rename = dict(postchange_start='start', prechange_end='end', radar_type='radar', site_lat='lat',
                  site_lon='lon',location='locn',short_name='name')
    cols_to_print = ['id', 'short_name','location', 'site_lon', 'site_lat', 'radar_type', 'postchange_start', 'prechange_end']
    print_df = df.loc[:, cols_to_print]
    print_df = print_df.rename(columns=rename)  # shorten some names
    with pd.option_context("display.max_columns", 10, 'display.max_colwidth', None, 'display.precision', 3,'display.width',200):
        print(print_df)
        print("="*70)

pd.options.mode.copy_on_write = True



radar_stn_data= ausLib.read_radar_file("meta_data/radar_site_list.csv")
# fill the dates
for c,d in zip(['postchange_start','prechange_end'],['1997-01-01','2024-01-01']):
    radar_stn_data[c]=radar_stn_data[c].fillna(d)


long_radar_data = radar_stn_data
# remove ALL WF44 & WF100.
radar = long_radar_data.radar_type
ok = ~(radar.str.startswith('WF44') | radar.str.startswith('WF100'))
print('Stations being dropped for radar check are: ',long_radar_data[~ok].short_name.values)
long_radar_data=long_radar_data[ok]
tmp = long_radar_data.loc[:,['id','postchange_start','prechange_end']]

# work out current stations...
current = tmp.groupby('id').prechange_end.max() > '2020-01-01'
# and old
old = tmp.groupby('id').postchange_start.min()  < '2005-01-01'
ids_ok = current.index.where(current & old ).dropna().astype('int')
print('Stations being dropped for date check are: ',
      long_radar_data[~long_radar_data.id.isin(ids_ok)].short_name.values)
long_radar_data = long_radar_data[long_radar_data.id.isin(ids_ok)]#.)

# now find cases with enough files...

my_logger = ausLib.setup_log(1)
file_count=dict()

for stn_id in long_radar_data.id.unique():
    name=long_radar_data.query(f'id=={stn_id}').head(1).short_name.values[0]
    root_dir = pathlib.Path(f'/g/data/rq0/hist_gndrefl/{stn_id}')
    n_files = len(list(root_dir.glob('*/*.zip')))
    file_count[stn_id] = n_files
    my_logger.info(f"ID: {stn_id} {name} has {n_files} files")

file_count=pd.Series(file_count,name='file_count')
ids=file_count[file_count>6000].index.unique()
long_radar_data = long_radar_data[long_radar_data.id.isin(ids)]#.)

# save results
long_radar_data.to_csv("meta_data/long_radar_stns.csv",date_format='%d/%m/%Y')
## print out result
print_df(long_radar_data)
