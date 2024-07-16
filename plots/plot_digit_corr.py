# plot the digitisation and bias correction from the DJF means.
import ausLib
import matplotlib.pyplot as plt
import xarray
import commonLib
import numpy as np
# get in the data first.
my_logger = ausLib.setup_log(2)
use_cache=True # set to false to reload data.
if not  (use_cache and 'metadata_digit_corr' in locals()):
    my_logger.info('Loading data.')
    metadata_digit_corr=dict()

    for site in ausLib.site_numbers.keys():
        meta_data_files = sorted((ausLib.data_dir / f'site_data/{site}_metadata/').glob(f'{site}_*_metadata.nc'))
        # need to check for existence to stop file system errors..
        ok = [f.exists() for f in meta_data_files]
        if not all(ok):
            FileNotFoundError(f'Not all files exist for {site}')

        ds = xarray.open_mfdataset(meta_data_files,concat_dim='time',combine='nested')
        ds = ds.sortby('time')
        metadata_digit_corr[site] =ds.load()
        my_logger.debug(f'Loaded data for {site}')
    radar_stns = ausLib.read_radar_file('meta_data/long_radar_stns.csv')
    my_logger.info('Loaded all data')

## now to plot the data.
fig, axes = ausLib.std_fig_axs(f'digit_corr', sharey=True,sharex=True)
for site, ax in axes.items():
    calibration_offset = metadata_digit_corr[site].calibration_offset
    calibration_offset.plot(ax=ax, color='k',ds='steps-pre')

    ax.set_title(site)
    ax.set_xlabel('Time')
    ax.set_ylabel('Calibration Corr (dBZ)',size='small')
    ax.tick_params(axis='x', labelsize='small',rotation=45) # set small font!
    ax.set_xlabel('Year')
    ax.label_outer()
    # plot TRMM orbit boost.
    ax.axvline(np.datetime64('2001-08-24'),color='blue',linestyle='--') # boost
    # and end of mission of 2014
    ax.axvline(np.datetime64('2014-07-15'),color='blue',linestyle='--') # out of fuel
    # add on meta-data to show when things change,.
    records = radar_stns.loc[radar_stns['id'] == ausLib.site_numbers[site]]
    lim = np.array(axes[site].get_ylim())
    y = 0.75 * lim[1] + 0.25 * lim[0]
    for name, r in records.iterrows():
        x = np.datetime64(r['postchange_start'])
        axes[site].axvline(x, color='red', linestyle='--')
        axes[site].text(x, y, r.radar_type, ha='left', va='center', fontsize='x-small', rotation=90)
    # add second axis for digitisation
    ax2 = ax.twinx()
    fraction = metadata_digit_corr[site].rapic_VIDRES
    fraction.plot(ax=ax2,drawstyle='steps-pre',color='purple')
    ax2.set_ylabel('Radar Resolution (bits)',color='purple',size='small')
    ax2.set_yscale('log')
    ticks = [8, 16, 32, 64, 128]
    # Set custom ticks and labels
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([str(tick) for tick in ticks],size='small',color='purple')
    ax2.set_title('')
    ax2.tick_params(axis='y', labelcolor='purple',which='minor',right=False)
    ax2.label_outer()
    #ax2.set_ylim(0.75,1.05)

fig.suptitle(f'Correction and Digitisation')
fig.show()
commonLib.saveFig(fig)
