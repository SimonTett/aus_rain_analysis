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
use_cache=True # set to false to reload data.
end_name_sens = '_rain_brisbane'  # calibration
end_name = '_rain_melbourne'

if not (use_cache and 'run_hr_day_dist' in locals()):
    my_logger.info('Loading data.')
    time_day=dict()
    time_day_sens=dict()
    day_of_season=dict()
    prd='1h'
    for site in ausLib.site_numbers.keys():
        name = site + end_name
        seas_mean = xarray.open_dataset(ausLib.data_dir / f'processed/{name}/seas_mean_{name}_DJF.nc')
        my_logger.info(f'Loaded {name}')
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
        run_hr_day_dist=True # loaded data

## now to plot them
fig, axs = ausLib.std_fig_axs(f'hr_day_dist', sharex=True, sharey=True)
for site, ax in axs.items():
    bins = np.arange(0, 24.5, 1)
    bins,edges,patches=time_day[site].plot.hist(bins=bins, color='k',
                                                density=True,ax=ax,histtype='step',linewidth=2)
    mode = int(edges[np.argmax(bins)])
    time_day_sens[site].plot.hist(bins=np.arange(0, 24.5, 1), density=True,
                                  ax=ax, histtype='step',color='k',linestyle='dashed')
    ax.set_title(site)
    ax.set_xlabel('Local Solar Hour',size='small')
    ticks=np.arange(0,25,3)
    labels=[str(tick) for tick in ticks]
    ax.set_xticks(ticks,labels=labels,size='small')
    #ax.tick_params(axis='x', )
    ax.set_ylabel('Probability Density',size='small')
    ax.text(0.1,0.8,f'Peak:{mode:2d}:00' ,transform=ax.transAxes,va='bottom',ha='left')
    ax.axhline(1.0/len(bins),color='k',linestyle='--') # what flat density would give
    ax.label_outer()
    ax2=ax.twiny()


    #ax2.set_ylim(0,0.2)
    step=5
    bins=np.arange(0,91,step)
    bins, edges, patches = day_of_season[site].to_xarray().plot.hist(bins=bins,  density=True, ax=ax2, histtype='step',
                                                    linewidth=2,color='purple')
    ticks=np.arange(0,91,30)
    labels=[str(tick) for tick in ticks]
    ax2.set_xticks(ticks,labels=labels,size='small',color='purple')
    #ax2.tick_params(axis='x', labelsize='small', labelcolor='purple')
    ax2.set_xlabel('Season Day',color='purple',size='small')
    ax2.axhline(1.0/(step*len(bins)),color='purple',linestyle='--') # what flat density would give
    #ax2.set_title('')
    ax2.label_outer()
fig.suptitle('Day Hour/Season Day')
fig.show()
commonLib.saveFig(fig)
commonLib.saveFig(fig,transpose=True)


