# plot the monthly-mean rain from radar
# Will plot all sites
import pathlib

import ausLib
import xarray
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import commonLib

calib = 'melbourne'

my_logger = ausLib.setup_log(1)

# read_plot_mean_monthly_rain = False # uncomment to force re-read of data
if not ('read_plot_mean_monthly_rain' in locals() and read_plot_mean_monthly_rain):
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
        mean_rain = ds.mean_raw_rain_rate.sel(x=slice(-5e3,5e3),y=slice(-5e3,5e3)).mean(['x','y']).load()
        total_rain[site] = mean_rain*mean_rain.time.dt.days_in_month*24
        #site_info[site] = ausLib.site_info(ausLib.site_numbers[site])
        gauge_total[site] = ds_obs.precip.sel(lon=site_rec.site_lon,lat=site_rec.site_lat,method='nearest').load()
        my_logger.info(f'Loaded data for {site}')
    read_plot_mean_monthly_rain = True # have read in data. Don't want to do it again!
    my_logger.info('Loaded all data')




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


