# Plot the ratio between the radar and gauge data as a function of range.
import pathlib

import ausLib
import xarray
import matplotlib.pyplot as plt
import commonLib
import numpy as np
import itertools

calib_colors = dict(melbourne='blue',brisbane='red')

my_logger = ausLib.setup_log(1)
if not ('read_plot_radar_guage_ratio_rain' in locals() and read_plot_radar_guage_ratio_rain):
    # taking advantage of lazy evaluation. if read_plot_mean_monthly_rain not been defined then won't check its value.
    # set false to force re-read
    r=125e3
    rgn = dict(x=slice(-r,r),y=slice(-r,r))
    bins = np.arange(0, r/1000.+5, 10)
    radar_rain_range = dict()
    gauge_rain_range = dict()
    radar_max_range = dict()
    radar_max = dict()
    site_info = dict()
    for calib,site in itertools.product(calib_colors.keys(),ausLib.site_numbers.keys()):

        direct = ausLib.data_dir / f'processed/{site}_rain_{calib}'
        file = direct/f'monthly_mean_{site}_rain_{calib}.nc'
        if not file.exists():
            my_logger.warning(f'No radar file {file}')
            continue

        ds = xarray.open_dataset(file)
        radar_rain = ds.mean_raw_rain_rate.sel(**rgn)
        radar_rain = radar_rain*ds.time.dt.days_in_month * 24 # monthly totals now
        radar_rain_max = ds.max_rain_rate.sel(resample_prd='1h',**rgn)
        r = np.sqrt(radar_rain.x ** 2 + radar_rain.y ** 2) / 1000.
        radar_rain['range'] = r
        radar_rain_max['range'] = r
        # read in the interpolated gauge data
        gauge_file = ausLib.data_dir / f'processed/{site}_rain_{calib}'/f'gauge_rain_{site}_{calib}.nc'
        if not gauge_file.exists():
            my_logger.warning(f'No gauge file {gauge_file}')
            continue
        gauge_rain = xarray.open_dataset(gauge_file).precip.sel(**rgn)
        r = np.sqrt(gauge_rain.x ** 2 + gauge_rain.y ** 2) / 1000.
        gauge_rain['range'] = r
        key = f'{site}_{calib}'
        radar_rain_range[key] = radar_rain.groupby_bins('range', bins=bins).mean().load()
        radar_max_range[key] = radar_rain_max.groupby_bins('range', bins=bins).median().load()
        radar_max[key] = radar_rain_max.median(['x','y']).load()
        gauge_rain_range[key] = gauge_rain.groupby_bins('range', bins=bins).mean().load()

        my_logger.info(f'Loaded gauge data for {calib} {site}')
        read_plot_radar_guage_ratio_rain=True


## now to plot the data
fig, axs = ausLib.std_fig_axs('radar_gauge_ratio_range',clear=True,sharex=True,sharey=True)
fig_max, axs_max = ausLib.std_fig_axs('radar_max_range',clear=True,sharex=True,sharey=True)
for name in radar_rain_range.keys():

    site,calib = name.split('_')
    ax = axs[site]
    ax_max = axs_max[site]
    L=radar_rain_range[name].time.dt.season=='DJF'
    L_gauge = gauge_rain_range[name].time.dt.season=='DJF'
    for time_range,linestyle in zip([slice('2015-12-01','2022-12-01'),slice('2007-12-01','2015-12-01')],['-','--']):
        time_label = f"{time_range.start[2:4]}-{time_range.stop[2:4]}"
        label = calib + '_' + time_label
        ratio = radar_rain_range[name][L].sel(time=time_range).mean('time')/gauge_rain_range[name][L].sel(time=time_range).mean('time')
        ratio.plot.line(x='range_bins',ax=ax,color=calib_colors[calib],label=label,linestyle=linestyle)
        mid_values =[v.mid for v in ratio.range_bins.values]
        max_iloc = ratio.argmax(...)
        x=mid_values[int(max_iloc['range_bins'])]
        ax.axvline(x, linestyle=linestyle, color='k')
        max_val = radar_max_range[name][L].sel(time=time_range).median('time')/radar_max[name][L].sel(time=time_range).median('time')
        max_val.plot.line(x='range_bins',ax=ax_max,color=calib_colors[calib],label=label,linestyle=linestyle)
        ax_max.axvline(x, linestyle=linestyle, color='k')

for site,ax in axs.items():
    ax.set_title(site)
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Radar/Gauge ratio')
    ax.axhline(1,linestyle='--',color='k')
    ax.set_xticks(np.arange(0,155,step=25))

for site,ax in axs_max.items():
    ax.set_title(site)
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('1h Rain (mm/h)')
    ax.set_xticks(np.arange(0,155,step=25))
    ax.axhline(1.0, linestyle='--', color='k')



axs['Adelaide'].legend()
fig.show()