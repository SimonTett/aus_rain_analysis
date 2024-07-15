# plot hour and area of max rain rate.
# will use the seas_mean data.
import matplotlib.pyplot as plt
import xarray
import ausLib
import scipy.stats
import numpy as np

import commonLib
import pathlib
import pandas as pd

my_logger = ausLib.setup_log(1)
end_name_sens = '_rain_brisbane'  # calibration
end_name = '_rain_melbourne'

time_day=dict()
time_day_sens=dict()
day_of_season=dict()
prd='1h'
for site in ausLib.site_numbers.keys():
    name = site + end_name
    seas_mean = xarray.open_dataset(ausLib.data_dir / f'processed/{name}/seas_mean_{name}_DJF.nc')
    lon = ausLib.site_info(ausLib.site_numbers[site]).iloc[0].loc['site_lon']
    t= seas_mean.sel(resample_prd=prd).time_max_rain_rate.stack(idx=['x', 'y', 'time']).dropna('idx').load()
    time_day[site] = (t.dt.hour + t.dt.minute / 60. +lon/15.)%24 # approx for local solar hour.
    dates = pd.to_datetime(t)
    season_year = dates.year - (dates.month != 12)
    season_start = pd.to_datetime(dict(year=season_year, month=12, day=1))

    day_of_season[site] = ((dates-season_start)/pd.Timedelta(1,'D')).astype(int)# days since 1st of december
    # sens case
    name = site + end_name_sens
    seas_mean = xarray.open_dataset(ausLib.data_dir / f'processed/{name}/seas_mean_{name}_DJF.nc')
    lon = ausLib.site_info(ausLib.site_numbers[site]).iloc[0].loc['site_lon']
    t= seas_mean.sel(resample_prd=prd).time_max_rain_rate.stack(idx=['x', 'y', 'time']).dropna('idx').load()
    time_day_sens[site] = (t.dt.hour + t.dt.minute / 60. +lon/15.)%24 # approx for local solar hour.
    my_logger.info(f'Loaded {name}')

## now to plot them
fig, axs = ausLib.std_fig_axs(f'day_hour_dist', sharex=True, sharey=True)
for site, ax in axs.items():
    bins,edges,patches=time_day[site].plot.hist(bins=np.arange(0, 24.5, 1), density=True,ax=ax,histtype='step',linewidth=2)
    mode = int(edges[np.argmax(bins)])
    time_day_sens[site].plot.hist(bins=np.arange(0, 24.5, 1), density=True, ax=ax, histtype='step',linewidth=2,color='red')
    ax.set_title(site )
    ax.set_xlabel('Local Solar Hour')
    ax.set_ylabel('Probability Density')
    ax.text(0.1,0.8,f'Peak:{mode:2d}:00' ,transform=ax.transAxes,va='bottom',ha='left')
    ax.label_outer()
    ax2=ax.twiny()
    ax2.set_xlabel('Day of Season',color='green')
    #ax2.set_ylim(0,0.2)
    bins, edges, patches = day_of_season[site].to_xarray().plot.hist(bins=np.arange(0,91,15),
                                                                     density=True, ax=ax2, histtype='step',
                                                    linewidth=2,color='green')
    ax2.set_title('')
    ax2.label_outer()
fig.suptitle('Hour of Day/Day of Season')
fig.show()
commonLib.saveFig(fig)


