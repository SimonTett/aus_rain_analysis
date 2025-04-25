# plot the sampling resolution and fraction of samples from the seasonal means.
import ausLib
import matplotlib.pyplot as plt
import xarray
import commonLib
import numpy as np
# get in the data first.
my_logger = ausLib.setup_log(1)
use_cache=True # set to false to reload data.
if not  (use_cache and 'datasets_resoln_fraction' in locals()):
    my_logger.info('Loading data.')
    conversion='_rain_melbourne'
    datasets_resoln_fraction=dict()

    for site in ausLib.site_numbers.keys():
        name = site + conversion
        event_file = ausLib.data_dir / f'processed/{name}/seas_mean_{name}_DJF.nc'
        if not event_file.exists():
            raise FileNotFoundError(f'No event file for {site} with {event_file}')
        datasets_resoln_fraction[site] = xarray.open_dataset(event_file)  # median extreme event
        my_logger.debug(f'Loaded {site} from {event_file}')
    radar_stns = ausLib.read_radar_file('meta_data/long_radar_stns.csv')
    my_logger.info('Loaded all data')

## now to plot the data.
fig, axes = ausLib.std_fig_axs(f'resoln_fraction', sharey=True,sharex=True,xtime=True)
for site, ax in axes.items():
    sample_resolution = datasets_resoln_fraction[site].sample_resolution.load()
    sample_resolution.dt.total_seconds().plot(ax=ax,
                                              color='k',marker='o',linestyle='None',ms=3)
    print(site, sample_resolution.time.max().dt.year.values)
    ax.set_title(site)
    ax.set_xlabel('Time')
    ax.set_ylabel('Sample Rate (s)',fontsize='small')
    ax.tick_params(axis='x', labelsize='small',rotation=45) # set small font!
    ax.tick_params(axis='y', labelsize='small') # set small font!
    ax.set_xlabel('Year')
    ax.label_outer()
    # add on meta-data to show when things change,.
    records = radar_stns.loc[radar_stns['id'] == ausLib.site_numbers[site]]
    lim = np.array(axes[site].get_ylim())
    y = 0.75 * lim[1] + 0.25 * lim[0]
    for name, r in records.iterrows():
        x = np.datetime64(r['postchange_start'])
        axes[site].axvline(x, color='red', linestyle='--')
        axes[site].text(x, y, r.radar_type, ha='left', va='center', fontsize='x-small', rotation=90)
    # add second axis for fraction
    ax2 = ax.twinx()
    fraction = datasets_resoln_fraction[site].fraction.load()
    fraction.plot(ax=ax2,drawstyle='steps-pre',marker='+',linestyle='None',ms=3,color='purple')
    ax2.set_ylabel('Fraction',color='purple',size='small')
    ticks=[0.75,0.8,0.85,0.9,0.95,1.0]
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([str(tick) for tick in ticks], size='small', color='purple')
    ax2.set_title('')
    ax2.label_outer()
    ax2.set_ylim(0.75,1.05)

fig.suptitle(f'Sampling resolution')
fig.show()
commonLib.saveFig(fig)
commonLib.saveFig(fig,transpose=True)
