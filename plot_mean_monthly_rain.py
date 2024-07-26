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
import cartopy.crs as ccrs





calib = 'melbourne'

my_logger = ausLib.setup_log(1)
sydney_sens_studies = ['Sydney_rain_brisbane','Sydney_rain_melbourne',
                 'Sydney_rain_melbourne_10min','Sydney_rain_melbourne_5_65']
#read_plot_mean_monthly_rain = False # uncomment to force re-read of data
if not ('read_plot_mean_monthly_rain' in locals() and read_plot_mean_monthly_rain):
    r=75e3
    rgn =  dict(x=slice(-r,r),y=slice(-r,r))
    obs_files = sorted(ausLib.agcd_rain_dir.glob('*.nc'))
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
        mean_rain = ds.mean_raw_rain_rate.sel(**rgn)
        total_rain[site] = (mean_rain.mean(['x','y'])*mean_rain.time.dt.days_in_month*24).load()
        #site_info[site] = ausLib.site_info(ausLib.site_numbers[site])
        # work  out region for gauge -- convert from m to long/l
        radar_proj = ausLib.radar_projection(ds.proj.attrs)
        trans_coords = ccrs.PlateCarree().transform_points(radar_proj,
                                                           x=np.array([-75e3,75e3]), y=np.array([-75e3,75e3]))
        # get in the gauge data then inteprolate to radar grid and mask by radar data
        lon_slice = slice(trans_coords[0,0],trans_coords[1,0])
        lat_slice = slice(trans_coords[0,1],trans_coords[1,1])
        gauge = ds_obs.precip.sel(lon=lon_slice,lat=lat_slice)
        longitude = mean_rain.longitude.isel(time=0).squeeze(drop=True)
        latitude= mean_rain.latitude.isel(time=0).squeeze(drop=True)
        time = mean_rain.time
        gauge = gauge.interp(lon=longitude, lat=latitude,time=time).where(mean_rain.notnull())
        gauge_total[site] = gauge.mean(['x','y']).load()
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
fig,axs = ausLib.std_fig_axs('monthly_mean_rain',sharex=True,sharey=True,clear=True)
fig2,axs2 = ausLib.std_fig_axs('monthly_mean_rain_ratio',sharex=True,sharey=True,clear=True)
roll_window =12
roll_window_ratio = 3
for site,ax in axs.items():
    #total_rain[site].plot(ax=ax,drawstyle='steps-mid',color='k')
    #gauge_total[site].plot(ax=ax,drawstyle='steps-mid',color='purple',alpha=0.5)
    total_rain[site].resample(time='QS-DEC').sum().plot(ax=ax,drawstyle='steps-post',color='blue')
    gauge_total[site].resample(time='QS-DEC').sum().plot(ax=ax,drawstyle='steps-post',color='purple')
    ax.set_ylabel('Total  (mm)',size='small')

    ax.set_title(site)
    ax.tick_params(axis='x', labelsize='small',rotation=45) # set small font!
    ax.set_ylim(10, None)

    # plot ratio
    ax2 = axs2[site]
    gt  = gauge_total[site].rolling(time=roll_window_ratio,center=True).mean()
    rt = total_rain[site].rolling(time=roll_window_ratio,center=True).mean()
    gt = gt.interp_like(rt)
    ratio = rt/gt
    ratio.plot(ax=ax2,drawstyle='steps-post',color='blue')

    ax2.set_title(f'Radar/Gauge {site}',size='small')
    ax2.set_ylabel('Ratio',size='small')
    ax2.tick_params(axis='x', labelsize='small',rotation=45) # set small font!
    ax2.axhline(1.0,linestyle='dashed',color='k')
    ax2.set_ylim(0.05,None)
    for a in [ax,ax2]:
        a.set_yscale('log')
        ausLib.plot_radar_change(a, site_info[site],trmm=True)

fig.suptitle('Seasonal mean rainfall')
fig2.suptitle('3-month rolling Radar/Gauge')
fig.show()
fig2.show()
fig_dir = pathlib.Path('extra_figs')
for f in [fig,fig2]:
    commonLib.saveFig(f,savedir=fig_dir)
    commonLib.saveFig(f,savedir=fig_dir,transpose=True)

# now to plot the Sydney sensitivity studies
fig,(ax,ax_ratio) = plt.subplots(1,2,figsize=(8,6),num='sydney_ss_mean_monthly_rain',
                       sharex=True,clear=True,layout='constrained')
gauge=gauge_total['Sydney'].resample(time='QS-DEC').sum()
gauge_rolling = gauge.rolling(time=roll_window_ratio,center=True).mean()
for file,col in zip(sydney_sens_studies,['red','blue','green','brown']):
    rm=total_rain[file].resample(time='QS-DEC').sum()
    rm.plot(ax=ax,drawstyle='steps-post',
            color=col,label=file.replace('Sydney_rain_',''))
    rm = total_rain[file].rolling(time=roll_window_ratio,center=True).mean()
    ratio = rm/gauge.interp_like(rm)
    ratio.plot(ax=ax_ratio,drawstyle='steps-post',color=col,label=file)
gauge.plot(ax=ax,label='AGCD gauge',drawstyle='steps-mid',color='purple')
ax.set_ylabel('Total  (mm)',size='small')
ax.set_title('Sydney total rainfall')
ax.set_ylim(10, 1000)
ax_ratio.set_title('Radar/Gauge Sydney')
ax_ratio.axhline(1.0, linestyle='dashed', color='k')
ax_ratio.set_ylim(0.05, 10)
for a in [ax,ax_ratio]:
    a.set_xlabel('Time')
    a.tick_params(axis='x', labelsize='small',rotation=45) # set small font!
    a.set_yscale('log')
    ausLib.plot_radar_change(a,site_info['Sydney'],trmm=True)


ax.legend()
fig.show()
commonLib.saveFig(fig,savedir=fig_dir)





