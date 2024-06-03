# calibrate radar data. Code needs cleaning up.
import pathlib

import ausLib
import xarray
import pathlib
import numpy as np
import cartopy.crs as ccrs
import cartopy.geodesic
import pandas as pd
import matplotlib.pyplot as plt
my_logger = ausLib.my_logger
ausLib.init_log(my_logger, 'DEBUG')

site = 'Melbourne'

in_radar = list(pathlib.Path(f'/scratch/wq02/st7295/summary_reflectivity/{site}').glob("hist_gndrefl*.nc"))
radar = xarray.open_mfdataset(in_radar)
max_ref =radar.max_reflectivity.sel(resample_prd='1h').where(radar.time.dt.season=='DJF', drop=True)
radar_coords=np.array([radar.proj.attrs['longitude_of_central_meridian'],
                  radar.proj.attrs['latitude_of_projection_origin']])
kw_proj=radar.proj.attrs.copy()
kw_proj['central_longitude']=kw_proj.pop('longitude_of_central_meridian')
kw_proj['central_latitude']=kw_proj.pop('latitude_of_projection_origin')
kw_proj['standard_parallels']=kw_proj.pop('standard_parallel').tolist()
for k in ['semi_major_axis', 'semi_minor_axis', 'grid_mapping_name']:
    kw_proj.pop(k, None)
proj_radar = ccrs.AlbersEqualArea(**kw_proj)
X,Y = np.meshgrid(radar.x.values, radar.y.values)
coords = ccrs.PlateCarree().transform_points(src_crs=proj_radar, x=X, y=Y)
radar_lon = coords[:, :, 0]
radar_lat = coords[:, :, 1]

# need to work out the projection for the radar data.
all_metadata = ausLib.read_gsdr_csv("AU_GSDR_metadata.csv")
yr_durn = (all_metadata.End_time - all_metadata.Start_time).dt.total_seconds() / (365.24 * 24 * 60 * 60)
all_metadata = all_metadata[yr_durn > 5]
L=all_metadata.End_time.dt.year >= 1997
all_metadata = all_metadata[L]
earth_geo = cartopy.geodesic.Geodesic()
pts = np.array([all_metadata.Longitude, all_metadata.Latitude]).T
dist = earth_geo.inverse(radar_coords, pts)
L = (dist[:, 0] < 75e3) & (all_metadata.End_time > '2010')  # closer than 75 km and data ending > 2010
radar_metadata = all_metadata[L]  # extract to 100 km radius.
gauge = dict()
radar_match = dict()
process_gauge = dict()
my_logger.info("Extracting gauge data")
lon_pts = []
lat_pts = []
name_pts = []
for name, series in radar_metadata.iterrows():
    gauge_data = ausLib.read_gsdr_data(series)
    # extract to DJF and from 2001 to 2015
    gauge_data = gauge_data['2000-12-01':'2015-02-28']
    L = gauge_data.index.month.isin([12, 1, 2])
    gauge_data = gauge_data[L]
    if (
    ~gauge_data.isnull()).sum() < 5 * 90 * 24:  # at least 5 summers worth of data (5 years * 90 days * 24 hours)
        continue
    process_gauge[name] = ausLib.process_gsdr_record(gauge_data, 'MS')
    L = process_gauge[name].index.month.isin([12, 1, 2])
    process_gauge[name] = process_gauge[name][L]
    gauge[name] = gauge_data
    lon_pts.append(series.Longitude)
    lat_pts.append(series.Latitude)
    name_pts.append(name)

if len(gauge) == 0:  # no comparison possible
   raise  ValueError(f"No gauges close enough for {site}. Skipping further processing")

my_logger.info("Extracting pts from radar")
guague_pts = np.row_stack((lon_pts, lat_pts))
xind, yind = ausLib.index_ll_pts(radar_lon, radar_lat, guague_pts,
                                 tolerance=2e-2)  # want to find to roughly 2 km.
xind = xarray.DataArray(xind, dims='station', coords=dict(station=list(gauge.keys())))
yind = xarray.DataArray(yind, dims='station', coords=dict(station=list(gauge.keys())))
radar_match = max_ref.sel(time=slice('2000-12-01', '2015-02-28')).isel(x=xind,y=yind).to_dataframe()
radar_match = radar_match.unstack('station').loc[:, 'max_reflectivity']
# and then make the time UTC.

# and fix the times.
radar_match.index =radar_match.index.to_period('M').to_timestamp()
radar_match.index = pd.to_datetime(radar_match.index, utc=True)
print(f"{site} has {len(radar_match.columns)} gauges to compare with")
## extract the station info.
lng = pd.Series({k: radar_metadata.loc[k].Longitude for k in process_gauge.keys()}).rename('longitude')
latitude = pd.Series({k: radar_metadata.loc[k].Latitude for k in process_gauge.keys()}).rename('latitude')
all_df=[]
for key, df in process_gauge.items():
    merge_df = pd.concat([df.max_rain.rename(f'gauge'), radar_match.loc[:, key].rename('radar_reflect')], axis=1)
    all_df.append(merge_df)
all_df = pd.concat(all_df, axis=0).reset_index(drop=True)
## now to do the stat fits
from statsmodels.api import RLM
formula = 'np.log(gauge) ~ np.log(radar_reflect)'
min_fit=1.
mx_mx = all_df.loc[:, 'radar_reflect'].max()*1.1
L = all_df.loc[:, 'gauge'] > min_fit # at least 1mm/hr max...
rfit = RLM.from_formula(formula, data=all_df[L]).fit()
radar_predict= pd.Series(np.geomspace(min_fit,mx_mx, 100)).rename('radar_reflect')
gauge_predict= np.exp(rfit.predict(radar_predict))
# plot things
plt.figure(num='robust_gauge_reflect',clear=True)
scatter_ax=plt.gca()
all_df[L].plot.scatter(x='gauge',y='radar_reflect', s=4,ax=scatter_ax)
scatter_ax.plot(gauge_predict,radar_predict,color='k',linewidth=2,linestyle='dashed')

scatter_ax.set_xlabel("Gauge Monthly Rx1h (mm/h)", fontsize='small')
scatter_ax.set_ylabel("Radar Reflect Rx1h (mm$^6$/m$^3$)", fontsize='small')
scatter_ax.set_title("Monthly Rx1h")

scatter_ax.set_yscale('log')
scatter_ax.set_xscale('log')


