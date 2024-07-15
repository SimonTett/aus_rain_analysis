# plot the digitisation and bias correction from the DJF means.
import ausLib
import matplotlib.pyplot as plt
import xarray
import commonLib
import numpy as np
# get in the data first.
my_logger = ausLib.setup_log(2)
metadata=dict()

for site in ausLib.site_numbers.keys():
    meta_data_files = sorted((ausLib.data_dir / f'site_data/{site}_metadata/').glob(f'{site}_*_metadata.nc'))
    ds = xarray.open_mfdataset(meta_data_files,concat_dim='time',combine='nested',parallel=True)
    ds = ds.sortby('time')
    ds = ds.resample(time='QS-DEC').mean()
    ds = ds.where(ds.time.dt.season == 'DJF',drop=True)
    metadata[site] =ds.load()
    my_logger.debug(f'Loaded data for {site}')
radar_stns = ausLib.read_radar_file('meta_data/long_radar_stns.csv')
my_logger.info('Loaded all data')

## now to plot the data.
fig, axes = ausLib.std_fig_axs(f'digit_corr', sharey=True,sharex=True)
for site, ax in axes.items():
    calibration_offset = metadata[site].calibration_offset
    calibration_offset.plot(ax=ax, color='k',ds='steps-pre')

    ax.set_title(site)
    ax.set_xlabel('Time')
    ax.set_ylabel('Calib Offset (dBZ)')
    ax.tick_params(axis='x', labelsize='small',rotation=45) # set small font!
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
    fraction = metadata[site].rapic_VIDRES
    fraction.plot(ax=ax2,drawstyle='steps-pre',color='purple')
    ax2.set_ylabel('Radar Res (bits)',color='purple')
    ax2.set_yscale('log')
    ticks = [8, 16, 32, 64, 128]
    # Set custom ticks and labels
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([str(tick) for tick in ticks])
    ax2.set_title('')
    ax2.label_outer()
    #ax2.set_ylim(0.75,1.05)

fig.suptitle(f'Corr and Digit')
fig.show()
commonLib.saveFig(fig)
