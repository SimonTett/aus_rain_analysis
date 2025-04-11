# plot the digitisation and bias correction from the DJF means.
import pathlib

import ausLib
import matplotlib.pyplot as plt
import xarray
import commonLib
import numpy as np

# get in the data first.
my_logger = ausLib.setup_log(2)
use_cache = True  # set to false to reload data.
if not (use_cache and 'metadata_digit_corr' in locals()):
    my_logger.info('Loading data.')
    metadata_digit_corr = dict()

    for site in ausLib.site_numbers.keys():
        meta_data_files = sorted((ausLib.data_dir / f'site_data/{site}_metadata/').glob(f'{site}_*_metadata.nc'))
        # need to check for existence to stop file system errors..
        ok = [f.exists() for f in meta_data_files]
        if not all(ok):
            FileNotFoundError(f'Not all files exist for {site}')

        ds = xarray.open_mfdataset(meta_data_files, concat_dim='time', combine='nested')
        ds = ds.sortby('time')
        metadata_digit_corr[site] = ds.load()
        my_logger.debug(f'Loaded data for {site}')
    radar_stns = ausLib.read_radar_file('meta_data/long_radar_stns.csv')
    my_logger.info('Loaded all data')

## now to plot the data.
fig, axes = ausLib.std_fig_axs(f'digit_corr', sharey=True, sharex=True)
fig2015, axes2015 = ausLib.std_fig_axs(f'corr_2015on', sharey=True, sharex=True)
for site, ax in axes.items():
    ax2 = axes2015[site]
    calibration_offset = metadata_digit_corr[site].calibration_offset
    calibration_offset.plot(ax=ax, color='k', ds='steps-pre')
    calibration_offset.sel(time=slice('2015-01-01', '2022-12-31')).plot(ax=ax2, color='k', ds='steps-pre')
    for a in [ax, ax2]:
        a.set_title(site)
        a.set_xlabel('Time')
        a.set_ylabel('Calibration Corr (dBZ)', size='small')
        a.tick_params(axis='x', labelsize='small', rotation=45)

    ausLib.plot_radar_change(ax, radar_stns.loc[radar_stns['id'] == ausLib.site_numbers[site]],trmm=True)
    # add second axis for digitisation
    ax2 = ax.twinx()
    fraction = metadata_digit_corr[site].rapic_VIDRES
    fraction.plot(ax=ax2, drawstyle='steps-pre', color='purple')
    ax2.set_ylabel('Radar Resolution (bits)', color='purple', size='small')
    ax2.set_yscale('log')
    ticks = [8, 16, 32, 64, 128]
    # Set custom ticks and labels
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([str(tick) for tick in ticks], size='small', color='purple')
    ax2.set_title('')
    ax2.tick_params(axis='y', labelcolor='purple', which='minor', right=False)
    ax2.label_outer()
    #ax2.set_ylim(0.75,1.05)

fig.suptitle(f'Correction and Digitisation')
fig.show()
fig2015.suptitle(f'Correction 2015 on')
fig2015.show()
commonLib.saveFig(fig)
commonLib.saveFig(fig, transpose=True)
commonLib.saveFig(fig2015, savedir=pathlib.Path('extra_figs'))
