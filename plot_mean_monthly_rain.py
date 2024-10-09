# plot the monthly-mean rain from radar
# Will plot all sites
import pathlib

import ausLib
import xarray
import matplotlib.pyplot as plt


import commonLib






calib = 'brisbane'

my_logger = ausLib.setup_log(1)
sydney_sens_studies = ['Sydney_rain_brisbane','Sydney_rain_melbourne',
                 'Sydney_rain_melbourne_10min','Sydney_rain_melbourne_5_65']
#read_plot_mean_monthly_rain = False # uncomment to force re-read of data
if not ('read_plot_mean_monthly_rain' in locals() and read_plot_mean_monthly_rain):
    # taking advantage of lazy evaluation. if read_plot_mean_monthly_rain not been defined then won't check its value.
    # set false to force re-read
    r=75e3
    rgn = dict(x=slice(-r,r),y=slice(-r,r))

    radar_total = dict()
    gauge_total = dict()
    site_info = dict()

    for site,no in ausLib.site_numbers.items():
        site_info[site] = ausLib.site_info(no)
        site_rec = site_info[site].iloc[0]
        direct = ausLib.data_dir / f'processed/{site}_rain_{calib}'
        file = direct/f'monthly_mean_{site}_rain_{calib}.nc'
        if not file.exists():
            my_logger.warning(f'No radar file {file}')
            continue

        ds = xarray.open_dataset(file)
        mean_rain = ds.mean_raw_rain_rate.sel(**rgn)
        radar_total[site] = (mean_rain.mean(['x', 'y']) * mean_rain.time.dt.days_in_month * 24).load()
        # read in the interpolated gauge data
        gauge_file = ausLib.data_dir / f'processed/{site}_rain_{calib}'/f'gauge_rain_{site}_{calib}.nc'
        if not gauge_file.exists():
            my_logger.warning(f'No gauge file {gauge_file}')
            continue
        gauge = xarray.open_dataset(gauge_file).precip.sel(**rgn)
        gauge_total[site] = gauge.mean(['x','y']).load()
        my_logger.info(f'Loaded gauge data for {site}')

    # read in the Sydney sens studies.
    for file in sydney_sens_studies:
        continue # skipp processing sysdney sens data
        direct = ausLib.data_dir / f'processed/{file}'
        filep = direct/f'monthly_mean_{file}.nc'
        if not filep.exists():
            raise FileNotFoundError(f'No file {filep}')
        ds = xarray.open_dataset(filep)

        mean_rain = ds.mean_raw_rain_rate.sel(**rgn).mean(['x','y']).load()
        radar_total[file] = mean_rain * mean_rain.time.dt.days_in_month * 24

        my_logger.info(f'Loaded data for {file}')
    read_plot_mean_monthly_rain = True # have read in data. Don't want to do it again!
    my_logger.info('Loaded all data')
else:
    my_logger.info('Data already loaded')




## now to plot data
fig,axs = ausLib.std_fig_axs(f'monthly_mean_rain_{calib}',sharex=True,sharey=True,clear=True)
fig2,axs2 = ausLib.std_fig_axs(f'monthly_mean_rain_ratio_{calib}',sharex=True,sharey=True,clear=True)
roll_window =12
roll_window_ratio = 3
for site,ax in axs.items():
    ax.tick_params(axis='x', labelsize='small',rotation=45) # set small font!
    ax.set_ylim(10, 1000)
    ax2 = axs2[site]
    ax2.set_ylabel('Ratio',size='small')
    ax2.tick_params(axis='x', labelsize='small',rotation=45) # set small font!
    if site not in radar_total:
        my_logger.warning(f'No radar data for {site}')
        continue
    if site not in gauge_total:
        my_logger.warning(f'No gauge data for {site}')
        continue

    radar_total[site].rolling(time=roll_window, center=True).mean().plot(ax=ax, color='blue')
    gauge_total[site].rolling(time=roll_window,center=True).mean().plot(ax=ax,color='purple')
    ax.set_ylabel('Total  (mm)',size='small')

    ax.set_title(site)



    # plot ratio

    gt  = gauge_total[site].rolling(time=roll_window_ratio,center=True).mean()
    rt = radar_total[site].rolling(time=roll_window_ratio, center=True).mean()
    gt = gt.interp_like(rt)
    ratio = rt/gt.where(gt > 30)
    ratio.plot(ax=ax2,drawstyle='steps-post',color='blue')

    ax2.set_title(f'Radar/Gauge {site}',size='small')

    ax2.axhline(1.0,linestyle='dashed',color='k')
    ax2.set_ylim(0.2,5)
    for a in [ax,ax2]:
        a.set_yscale('log')
        from matplotlib.ticker import LogLocator, ScalarFormatter
        a.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0]))
        a.yaxis.set_major_formatter(ScalarFormatter())
        a.tick_params(axis='y', labelsize='small', rotation=45)
        ausLib.plot_radar_change(a, site_info[site],trmm=True)

fig.suptitle('12-month  rolling mean rainfall')
fig2.suptitle('3-month rolling Radar/Gauge')
fig.show()
fig2.show()
fig_dir = pathlib.Path('extra_figs')
fig_type = ['png','pdf']
for f in [fig,fig2]:
    commonLib.saveFig(f,savedir=fig_dir,figtype=fig_type)
    commonLib.saveFig(f,savedir=fig_dir,transpose=True,figtype=fig_type)


# now to plot the Sydney sensitivity studies
fig,(ax,ax_ratio) = plt.subplots(1,2,figsize=(8,6),num='sydney_ss_mean_monthly_rain',
                       sharex=True,clear=True,layout='constrained')
gauge=gauge_total['Sydney'].resample(time='QS-DEC').sum()
gauge_rolling = gauge.rolling(time=roll_window_ratio,center=True).mean()
for file,col in zip(sydney_sens_studies,['red','blue','green','brown']):
    rm=radar_total[file].resample(time='QS-DEC').sum()
    rm.plot(ax=ax,drawstyle='steps-post',
            color=col,label=file.replace('Sydney_rain_',''))
    rm = radar_total[file].rolling(time=roll_window_ratio, center=True).mean()
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





