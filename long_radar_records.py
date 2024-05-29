# find the radar stations that have long records. Print out relevant meta-data
import pandas as pd
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns

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

file_data = pd.read_csv("file_count.txt",header=None).rename(columns={0:'root_dir',1:'file_count'})
long_dirs = file_data[file_data.file_count > 6500].root_dir # remove this
ids = long_dirs.str.split("/",expand=True).iloc[:,5].astype('int')

radar_stn_data= pd.read_csv("meta_data/radar_site_list.csv")
L=radar_stn_data.id.isin(ids)
max_Time_str = pd.to_datetime(radar_stn_data.prechange_end.str.replace('-','1/1/2000'),dayfirst=True).max().strftime('%d/%m/%Y')
long_radar_data = radar_stn_data[L]
# remove ALL WF44 & WF100.
radar = long_radar_data.radar_type
ok = ~(radar.str.startswith('WF44') | radar.str.startswith('WF100'))
long_radar_data=long_radar_data[ok]
tmp = long_radar_data.loc[:,['id','postchange_start','prechange_end']]
for k in ['postchange_start','prechange_end']:
    tmp[k]=pd.to_datetime(tmp[k].str.replace("-",max_Time_str),dayfirst=True)
# work out current stations...
current = tmp.groupby('id').prechange_end.max() > '2022-01-01'
# and old
old = tmp.groupby('id').postchange_start.min()  < '2005-01-01'
ids_ok = current.index.where(current & old ).dropna().astype('int')
long_radar_data = long_radar_data[long_radar_data.id.isin(ids_ok)]#.)
# save results
long_radar_data.to_csv("meta_data/long_radar_stns.csv")
## print out result
print_df(long_radar_data)


## make a map of the long lasting radar stations

fig = plt.figure(num="long_radar_aus",figsize=(8,5),clear=True,layout='constrained')
ax = fig.add_subplot(111,projection=ccrs.PlateCarree())
ax.set_extent([110,160,-45,-10])
ax.coastlines()
lr=long_radar_data.groupby('id').tail(1) # most recent records for each ID
lr = lr.sort_values('id',axis=0).set_index('id',drop=False)

# add in changes.
changes = long_radar_data.id.value_counts().rename("changes")
lr = lr.join(changes)

sns.scatterplot(data=lr,x='site_lon',y='site_lat',hue='radar_type',style='beamwidth',
                ax=ax,sizes=(40,80),size='changes',markers=['o','s','h'])
#lr.plot.scatter(x='site_lon',y='site_lat',marker='o',ax=ax,s=20)
for name,row in lr.iterrows():
    ax.text(row.site_lon,row.site_lat,row.short_name,va='bottom')

ax.set_title("Australian radars with > 6500 obs days")
fig.show()
fig.savefig('figures/long_radar_aus.png')
print_df(lr)



