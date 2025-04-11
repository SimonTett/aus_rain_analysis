# plot the rain ratio vs  correction.
# Rain ratios are converted to DBz.
import pathlib

import ausLib
import matplotlib.pyplot as plt
import xarray
import commonLib
import numpy as np


# get in the data first.
calib='melbourne'
my_logger = ausLib.setup_log(2)
use_cache = True  # set to false to reload data.
if not (use_cache and 'correction' in locals()):
    my_logger.info('Loading data.')
    correction  = dict()
    rain_ratio = dict()
    site_info = dict()
    r=75e3
    time_rgn = slice('2000-01-01', '2022-12-31')
    rgn = dict(x=slice(-r,r),y=slice(-r,r),time=time_rgn)
    for site,no in ausLib.site_numbers.items():
        rolling =1
        meta_data_files = sorted((ausLib.data_dir / f'site_data/{site}_metadata/').glob(f'{site}_*_metadata.nc'))
        # need to check for existence to stop file system errors..
        ok = [f.exists() for f in meta_data_files]
        if not all(ok):
            FileNotFoundError(f'Not all files exist for {site}')
        for calib in ['melbourne','brisbane']:

            ds = xarray.open_mfdataset(meta_data_files, concat_dim='time', combine='nested')
            ds = ds.sortby('time').sel(time=time_rgn)
            if rolling > 1:
                correction[site] = ds.calibration_offset.rolling(time=rolling,center=True).mean().load()
            else:
                correction[site] = ds.calibration_offset.load()
            site_info[site] = ausLib.site_info(no)
            # get radar data
            direct = ausLib.data_dir / f'processed/{site}_rain_{calib}'
            file = direct/f'monthly_mean_{site}_rain_{calib}.nc'
            if not file.exists():
                my_logger.warning(f'No radar file {file}')
                continue

            ds = xarray.open_dataset(file)
            radar_rain = ds.mean_raw_rain_rate.sel(**rgn)
            radar_rain = (radar_rain.mean(['x', 'y']) * radar_rain.time.dt.days_in_month * 24).load()
            if rolling > 1:
                radar_rain = radar_rain.rolling(time=rolling, center=True).mean()
            # read in the interpolated gauge data
            gauge_file = ausLib.data_dir / f'processed/{site}_rain_{calib}'/f'gauge_rain_{site}_{calib}.nc'
            if not gauge_file.exists():
                my_logger.warning(f'No gauge file {gauge_file}')
                continue
            gauge_rain = xarray.open_dataset(gauge_file).precip.sel(**rgn)
            gauge_rain = gauge_rain.mean(['x','y']).load()
            if rolling > 1:
                gauge_rain = gauge_rain.rolling(time=rolling, center=True).mean()
            L=gauge_rain > 10.0 # wet enough  months
            gauge_rain = gauge_rain.where(L)
            rain_ratio[site+'_'+calib] = radar_rain/gauge_rain
        my_logger.info(f'Loaded data for {site}')

    my_logger.info('Loaded all data')

# plot the data
for calib in ['melbourne','brisbane']:
    fig, axes = ausLib.std_fig_axs(f'scatter_ratio_corr_{calib}',
                                   sharey=True, sharex=True, clear=True, layout='constrained')
    for site, ax in axes.items():
        rr = np.log10(rain_ratio[site+'_'+calib].interp(time=correction[site].time))*10

        # add on means grouped by correction
        rr_mean = rr.groupby(correction[site].round(0)).mean()
        rr_mean.plot(ax=ax, color='k', marker='+',ms=12,mew=1.5,linestyle='solid')
        rr_mean = rr.groupby(correction[site].round(1)).mean()
        rr_mean.plot(ax=ax, color='k', marker='+',ms=8,mew=1,linestyle='None')
        cmap=ax.scatter(correction[site], rr,marker='o',c=rr.time.dt.year,s=6,cmap='viridis',vmin=2000,vmax=2022,plotnonfinite=True)
        ax.set_title(site)
        ax.set_xlabel('Correction (dBZ)')
        ax.set_ylabel('Radar/Gauge (dBZ)')
        # ax.set_yscale('log')
        ax.label_outer()
        ax.axhline(0.0,color='k')
        ax.set_ylim(-8,8.)

    fig.colorbar(cmap, ax=axes.values(), orientation='horizontal', fraction=0.1, aspect=40, pad=0.05)
    fig.suptitle(f'Rain ratio vs  correction for {calib}')
    fig.show()
    commonLib.saveFig(fig,savedir=pathlib.Path('extra_figs'))
