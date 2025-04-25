# Plot the seasonal mean rainfall ratio between radar and gauge
# tricky part is masking the gauge data by radar data.
import pathlib

import pandas as pd

import ausLib
import xarray
import matplotlib.pyplot as plt


import commonLib


calib = 'melbourne'

my_logger = ausLib.setup_log(1)

#read_plot_mean_monthly_rain = False # uncomment to force re-read of data

def process_gauge(ds, site_name):
    # process the gauge data to get the total rainfall
    # mask out radar data
    radar = radar_total[site_name]
    # Convert gauge to seasonal totals.
    gauge = ds.precip.resample(time='QS-DEC').sum()
    L = gauge.time.dt.season == 'DJF'
    gauge = gauge.sel(time=L,drop=True)
    gauge = gauge.where(radar.notnull())
    return gauge
if not ('read_plot_mean_monthly_rain' in locals() and read_plot_mean_monthly_rain):
    # taking advantage of lazy evaluation. if read_plot_mean_monthly_rain not been defined then won't check its value.
    # set false to force re-read
    radar_total=dict()
    gauge_total=dict()
    gauge_file = lambda site_name: ausLib.data_dir / f'processed/{site_name}'/f'gauge_rain_{site_name.replace("_rain","")}.nc'
    for calib in ['melbourne','brisbane']:

        radar_total.update(ausLib.read_process(process=lambda ds,site_name: ds.mean_raw_rain_rate*90*24,
                                           conversion=f'_rain_{calib}'))
    # for comparison with gauge data convert to total rain


        gauge_total.update(ausLib.read_process(process=process_gauge,seas_file_fn=gauge_file,
                                               conversion=f'_rain_{calib}'))

    read_plot_mean_monthly_rain = True # have read in data. Don't want to do it again!
    my_logger.info('Loaded all data')
else:
    my_logger.info('Data already loaded')




## now to plot data
fig,axs = ausLib.std_fig_axs(f'DJF_rain_ratio',sharex=True,sharey=True,clear=True,xtime=True)

for site,ax in axs.items():
    ax.set_ylabel('Ratio',size='small')
    ax.tick_params(axis='x', labelsize='small',rotation=45) # set small font!
    for calib,color in zip(['melbourne','brisbane'],['black','purple']):
        name = site + '_rain_' + calib
        try:
            rt = radar_total[name].mean(['x', 'y'])
        except KeyError:
            my_logger.warning(f'No radar data for {name}')
            continue
        try:
            gt = gauge_total[name].mean(['x', 'y'])
            gt = gt.interp_like(rt)
        except KeyError:
            my_logger.warning(f'No gauge data for {name}')
            continue
    # plot ratio
        ratio = (rt/gt).where(gt > 5)
        ratio.plot(ax=ax,drawstyle='steps-post',color=color,linewidth=2,label=calib.capitalize())

    ax.set_title(f'{site}')

    ax.axhline(1.0,linestyle='dashed',color='k')
    ax.set_ylim(0.2,5)
    ax.set_ylabel('Radar/Gauge',size='small')
    ax.set_yscale('log')
    from matplotlib.ticker import LogLocator, ScalarFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis='y', labelsize='small', rotation=45)
    ausLib.plot_radar_change(ax, ausLib.site_info(site),trmm=True,site_start=True)
    ax.label_outer()
# put some legends on
handles, labels = axs['Melbourne'].get_legend_handles_labels()
fig.legend(handles, labels, ncol=1,fontsize='small',loc=(0.4, 0.9),handletextpad=0.2,handlelength=3.,columnspacing=1.0)

fig.show()
commonLib.saveFig(fig)
commonLib.saveFig(fig, transpose=True)





