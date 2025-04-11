# plot data for Adelaide
# show t/s data of median max rain from two calibrations
import pathlib

import ausLib
import xarray
import matplotlib.pyplot as plt
import numpy as np
import commonLib
my_logger = ausLib.setup_log(1)
site='Adelaide'
max_rain = dict()
mean_rain = dict()
resample='30min'
rgn = dict(x=slice(-75e3,75e3),y=slice(-75e3,75e3))
for calib in ['melbourne','brisbane']:
    file = ausLib.data_dir / f'processed/{site}_rain_{calib}'/f'monthly_mean_{site}_rain_{calib}.nc'
    if not file.exists():
        raise FileNotFoundError(f'No file {file}')

    ds = xarray.open_dataset(file)
    max_rain[calib] = ds.max_rain_rate.sel(resample_prd=resample,**rgn).load()
    mean_rain[calib] = ds.mean_raw_rain_rate.sel(**rgn).mean(['x','y']).load()
    mean_rain[calib] *= ds.time.dt.days_in_month * 24 # from mm/h to mm
    my_logger.info(f'Loaded data for {calib}')

# get in the gauge mean data
gauge_file = ausLib.data_dir / f'processed/{site}_rain_{calib}'/f'gauge_rain_{site}_{calib}.nc'
gauge = xarray.open_dataset(gauge_file).precip.sel(**rgn)
mean_rain['gauge'] = gauge.mean(['x','y']).load()

args = dict(num='Adelaide_detail', clear=True, figsize=(8, 6), layout='constrained',
            empty_sentinel='BLANK', )  # default args,
mosaic = [['mean','Seas_cycle'],['max_median','distribution']]
fig, axes = plt.subplot_mosaic(mosaic, **args)
bins=np.linspace(0,150)
colors=dict(melbourne='blue',brisbane='red',gauge='purple')


for calib in max_rain.keys():
    max_rain[calib].median(['x','y']).plot(ax=axes['max_median'], label=calib,color=colors[calib])
    max_rain[calib].plot.hist(ax=axes['distribution'], bins=bins, density=True,histtype='step', label=calib,yscale='log',color=colors[calib])
    max_rain[calib].sel(time='2015').plot.hist(ax=axes['distribution'], bins=bins, density=True, histtype='step', label='2015_'+calib,
                              yscale='log', color=colors[calib],linestyle='dashed')
for calib in mean_rain.keys():
    mean_rain[calib].plot(ax=axes['mean'],label=calib,color=colors[calib])
    # plot the seasonal cycle
    mean_rain[calib].groupby('time.month').mean().plot(ax=axes['Seas_cycle'],label=calib,color=colors[calib])
    # add on the 2015 data.
    mean_rain[calib].sel(time='2015').groupby('time.month').mean().plot(ax=axes['Seas_cycle'],label='2015'+calib,color=colors[calib],linestyle='dashed')


axes['mean'].set_title('Mean rain')
axes['mean'].set_ylabel('mm/month')
axes['Seas_cycle'].set_title('Seasonal cycle')
axes['Seas_cycle'].set_ylabel('mm/month')
axes['max_median'].set_title('Median max rain ')
axes['max_median'].set_ylabel('mm/hr')

axes['distribution'].set_title('Max rain dist.')
axes['distribution'].set_xlabel('mm/hr')
axes['distribution'].set_ylabel('Density')
axes['Seas_cycle'].legend(ncol=3,fontsize='small',handletextpad=0.2, handlelength=3.0, columnspacing=0.5)
for l in ['mean','max_median']:
    axes[l].tick_params(axis='x', labelsize='small', rotation=45)  # set small font!
    axes[l].set_ylim(0, None)
fig.show()
commonLib.saveFig(fig, savedir=pathlib.Path('extra_figs'),figtype='pdf')


