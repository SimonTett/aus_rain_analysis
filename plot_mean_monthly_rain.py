# plot the monthly-mean rain from radar
# Will plot all sites
import pathlib

import ausLib
import xarray
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import commonLib
import cartopy
import cartopy.geodesic

def obs_precip_radius(obs_precip:xarray.DataArray,lon:float,lat:float,
                      radius:np.ndarray) -> xarray.DataArray:
    pts = np.meshgrid(obs_precip.lon, obs_precip.lat)
    pts = np.vstack([pts[0].flatten(), pts[1].flatten()]).T
    earth_geo = cartopy.geodesic.Geodesic()
    dist = earth_geo.inverse([lon, lat], pts)
    coords = [obs_precip.lon, obs_precip.lat]
    rad = xarray.DataArray(dist[:, 0].reshape(len(coords[0]), len(coords[1])) / 1e3,
                           coords=coords)
    result = obs_precip.groupby_bins(rad, radius).mean()
    return result
# use as follows:
# dd=ds_obs.precip.sel(lon=slice(site_rec.site_lon-2,site_rec.sat_lon+2),
#   lat=site_rec.site_lat,method='nearest').load()
# p_r = obs_precip_radius(dd,site_rec.site_lon,site_rec.site_lat,np.arange(0,200,50))


calib = 'melbourne'

my_logger = ausLib.setup_log(1)
sydney_sens_studies = ['Sydney_rain_brisbane','Sydney_rain_melbourne',
                 'Sydney_rain_melbourne_10min','Sydney_rain_melbourne_5_65']
#read_plot_mean_monthly_rain = False # uncomment to force re-read of data
if not ('read_plot_mean_monthly_rain' in locals() and read_plot_mean_monthly_rain):
    r=75e3
    rgn =  dict(x=slice(-r,r),y=slice(-r,r))
    obs_files = sorted(ausLib.agcd_rain_dir.glob('*.nc')) # region to extract
    ds_obs = xarray.open_mfdataset(obs_files).sel(time=slice('1997', None))
    total_rain = dict()
    gauge_total = dict()
    site_info = dict()
    # taking advantage of lazy evaluation. if read_plot_mean_monthly_rain not been defined then won't check its value.
    # set false to force re-read
    for site,no in ausLib.site_numbers.items():
        site_info[site] = ausLib.site_info(no)
        site_rec = site_info[site].iloc[0]
        direct = ausLib.data_dir / f'processed/{site}_rain_{calib}'
        file = direct/f'monthly_mean_{site}_rain_{calib}.nc'
        if not file.exists():
            raise FileNotFoundError(f'No file {file}')

        ds = xarray.open_dataset(file)
        mean_rain = ds.mean_raw_rain_rate.sel(**rgn).mean(['x','y']).load()
        total_rain[site] = mean_rain*mean_rain.time.dt.days_in_month*24
        #site_info[site] = ausLib.site_info(ausLib.site_numbers[site])
        gauge_total[site] = ds_obs.precip.sel(lon=site_rec.site_lon,lat=site_rec.site_lat,method='nearest').load()
        my_logger.info(f'Loaded data for {site}')
    # read in the Sydney sens studies.
    for file in sydney_sens_studies:
        direct = ausLib.data_dir / f'processed/{file}'
        filep = direct/f'monthly_mean_{file}.nc'
        if not filep.exists():
            raise FileNotFoundError(f'No file {filep}')
        ds = xarray.open_dataset(filep)

        mean_rain = ds.mean_raw_rain_rate.sel(**rgn).mean(['x','y']).load()
        total_rain[file] = mean_rain*mean_rain.time.dt.days_in_month*24

        my_logger.info(f'Loaded data for {file}')
    read_plot_mean_monthly_rain = True # have read in data. Don't want to do it again!
    my_logger.info('Loaded all data')
else:
    my_logger.info('Data already loaded')




## now to plot data
fig,axs = ausLib.std_fig_axs('monthly_mean_rain',sharex=True,clear=True)
fig2,axs2 = ausLib.std_fig_axs('monthly_mean_rain_ratio',sharex=True,clear=True)
for site,ax in axs.items():
    #total_rain[site].plot(ax=ax,drawstyle='steps-mid',color='k')
    #gauge_total[site].plot(ax=ax,drawstyle='steps-mid',color='purple',alpha=0.5)
    total_rain[site].rolling(time=12,center=True).mean().plot(ax=ax,drawstyle='steps-mid',color='k')
    gauge_total[site].rolling(time=12,center=True).mean().plot(ax=ax,drawstyle='steps-mid',color='purple',alpha=0.5)
    ax.set_ylabel('Total  (mm)',size='small')
    ausLib.plot_radar_change(ax,site_info[site])
    ax.set_title(site)
    ax.tick_params(axis='x', labelsize='small',rotation=45) # set small font!
    # plot ratio
    ax2 = axs2[site]
    gt  = gauge_total[site].rolling(time=12,center=True).mean()
    rt = total_rain[site].rolling(time=12,center=True).mean()
    gt = gt.interp_like(rt)
    ratio = rt/gt
    ratio.plot(ax=ax2,drawstyle='steps-mid',color='k')
    ausLib.plot_radar_change(ax2,site_info[site])
    ax2.set_title(f'Radar/Gauge {site}',size='small')
    ax2.set_ylabel('Ratio',size='small')
    ax2.tick_params(axis='x', labelsize='small',rotation=45) # set small font!

fig.show()
fig2.show()
fig_dir = pathlib.Path('extra_figs')
for f in [fig,fig2]:
    commonLib.saveFig(f,savedir=fig_dir)
    commonLib.saveFig(f,savedir=fig_dir,transpose=True)

# now to plot the Sydney sensitivity studies
fig,(ax,ax_ratio) = plt.subplots(1,2,figsize=(8,6),num='sydney_ss_mean_monthly_rain',
                       sharex=True,clear=True,layout='constrained')
gauge=gauge_total['Sydney'].rolling(time=12,center=True).mean()
for file,col in zip(sydney_sens_studies,['red','blue','green','brown']):
    rm=total_rain[file].rolling(time=12,center=True).mean()
    rm.plot(ax=ax,drawstyle='steps-mid',color=col,label=file)
    ratio = rm/gauge.interp_like(rm)
    ratio.plot(ax=ax_ratio,drawstyle='steps-mid',color=col,label=file)
gauge.plot(ax=ax,label='AGCD gauge',drawstyle='steps-mid',color='purple',alpha=0.5)
ax.set_ylabel('Total  (mm)',size='small')
ax.set_title('Sydney total rainfall')
ax_ratio.set_title('Radar/Gauge Sydney')
for a in [ax,ax_ratio]:
    a.set_xlabel('Time')
    a.tick_params(axis='x', labelsize='small',rotation=45) # set small font!
    ausLib.plot_radar_change(a,site_info['Sydney'])

ax.legend()
fig.show()
commonLib.saveFig(fig,savedir=fig_dir)





