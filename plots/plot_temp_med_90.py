# plot the sampling resolution and fraction of samples from the seasonal means.
import ausLib
import matplotlib.pyplot as plt
import xarray
import commonLib
import numpy as np
import pandas as pd
from process_events import acorn_lookup # lookup that gives acorn ids for temperature sites
# get in the data first.
my_logger = ausLib.setup_log(1)
use_cache=True # set to false to reload data.
if not  (use_cache and 'loaded_temp_med_90' in locals()):
    my_logger.info('Loading data.')
    conversion='_rain_melbourne'
    seas_med=dict()
    seas_90=dict()
    temps=dict()

    for site in ausLib.site_numbers.keys():
        name = site + conversion
        seas_file = ausLib.data_dir / f'processed/{name}/seas_mean_{name}_DJF.nc'
        if not seas_file.exists():
            raise FileNotFoundError(f'No season  file for {site} with {seas_file}')
        seas_data = xarray.open_dataset(seas_file)  # median extreme event.
        seas_med[site] = seas_data.max_rain_rate.median(dim=['x','y']).load()
        seas_90[site] = seas_data.max_rain_rate.quantile(0.9,dim=['x','y']).drop_vars('quantile')
        station_id = acorn_lookup[site]
        obs_temperature = ausLib.read_acorn(station_id, what='mean').resample('QS-DEC').mean()
        # get times to middle of season.
        offset = (obs_temperature.index.diff() / 2.).fillna(pd.Timedelta(45, 'D'))
        obs_temperature.index = obs_temperature.index + offset
        attrs = obs_temperature.attrs.copy()
        obs_temperature = obs_temperature.to_xarray().rename('ObsT').rename(dict(date='time')).assign_attrs(attrs)
        L=obs_temperature.time.dt.season=='DJF'
        obs_temperature = obs_temperature.where(L).dropna('time')
        temps[site] = obs_temperature

        my_logger.debug(f'Loaded {site} from {seas_file}')
    radar_stns = ausLib.read_radar_file('meta_data/long_radar_stns.csv')
    my_logger.info('Loaded all data')
    loaded_temp_med_90= True

## now to plot the data.
fig, axes = ausLib.std_fig_axs(f'temp_med_90',sharex=True)
prd='1h'
ylim=(None,None)
rain_limits=dict(Mornington=(0,25),Grafton=(0,30),Canberra=(0,40),Adelaide=(0,40))
temp_limits=dict(Mornington=(10,30),Grafton=(10,30),Canberra=(10,40),Adelaide=(10,40))
for site, ax in axes.items():
    if site in rain_limits:
        ylim = rain_limits[site]
    seas_med[site].sel(resample_prd=prd).plot(ax=ax,color='blue',drawstyle='steps-pre',label='Median Max')
    seas_90[site].sel(resample_prd=prd).plot(ax=ax,color='red',drawstyle='steps-pre',label='90% Max')
    ax.set_title(site)
    ax.set_xlabel('Time')
    ax.set_ylabel('Rainfall (mm)',fontsize='small')
    ax.tick_params(axis='x', labelsize='small',rotation=45) # set small font!
    ax.tick_params(axis='y', labelsize='small') # set small font!
    ax.set_xlabel('Year')

    ax._label_outer_xaxis(skip_non_rectangular_axes=True)

    # add on meta-data to show when things change,.
    records = radar_stns.loc[radar_stns['id'] == ausLib.site_numbers[site]]
    lim = np.array(axes[site].get_ylim())
    y = 0.75 * lim[1] + 0.25 * lim[0]
    for name, r in records.iterrows():
        x = np.datetime64(r['postchange_start'])
        axes[site].axvline(x, color='k', linestyle='--')
        axes[site].text(x, y, r.radar_type, ha='left', va='center', fontsize='x-small', rotation=90)
    # add second axis for temperature
    ax2 = ax.twinx()
    temps[site].sel(time=slice('1997','2021')).plot(ax=ax2,drawstyle='steps-pre',linestyle='None',
                                                    marker='*',ms=5,color='purple',label='Local Temp')
    ax2.set_ylabel('',color='purple',size='x-small')
    ax2.tick_params(axis='y', labelsize='x-small',labelcolor='purple')


    ax2.set_title('')
    #ax2._label_outer_xaxis(skip_non_rectangular_axes=True)


handles, labels = axes['Melbourne'].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels() # last ax
handles += h2
labels += l2
for handle in handles:
    handle._sizes= [10.0] # ass suggested by chatGPT
legend=fig.legend(handles, labels, loc=(0.4, 0.9), fontsize='small')
fig.suptitle(f'Rx{prd} Med/90% Rainfall & Local DJF T')
fig.show()
commonLib.saveFig(fig)
commonLib.saveFig(fig,transpose=True)
