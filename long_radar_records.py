# find the radar stations that have long records. Print out relevant meta-data
import pandas as pd
file_data = pd.read_csv("file_count.txt",header=None).rename(columns={0:'root_dir',1:'file_count'})
long_dirs = file_data[file_data.file_count > 7000].root_dir
ids = long_dirs.str.split("/",expand=True).iloc[:,5].astype('int')

radar_stn_data= pd.read_csv("radar_site_list.csv")
L=radar_stn_data.id.isin(ids)
long_stn_data = radar_stn_data[L]
tmp = long_stn_data.loc[:,['id']]
tmp['prechange_end']=long_stn_data.prechange_end.str.replace("-",'31/12/2023')
# work out current stations...
current = pd.to_datetime(tmp.groupby('id').prechange_end.max(),dayfirst=True) > '2022-01-01'
ids_ok = current.index.where(current).dropna().astype('int')
long_stn_data = long_stn_data[long_stn_data.id.isin(ids_ok)].rename(columns=dict(postchange_start='start',prechange_end='end'))
## print out result
with pd.option_context("display.max_columns",10,'display.max_colwidth',160):
    print(long_stn_data.loc[:,['id','short_name','site_lon','site_lat','radar_type','start','end']])



