# plot sensitivity studies for Sydney.
# Calibration -- melbourne and brisabne + melbourne calibration with larger dbz range.

import ausLib
import xarray
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import commonLib
import numpy as np
from R_python import gev_r

my_logger = ausLib.setup_log(1)
seas_files = list((ausLib.data_dir / 'processed').glob('Sydney_rain*/seas_mean*_Sydney_rain_*_DJF.nc'))
seas_files = [f for f in seas_files if not ('check'  in str(f) or 'old' in str(f))]
seas_mean = {}
seas_gev_fit = dict()
gev_t_fit = dict()
gev_t_fit_bs = dict()
gev_fit = dict()
gev_fit_bs = dict()
rain_range = dict()
recreate_fit = False
for file in seas_files:
    name = file.parent.name.replace('Sydney_rain_', '').capitalize()
    da = xarray.open_dataset(file).max_rain_rate
    resample_hours = pd.to_timedelta(da.resample_prd) / pd.Timedelta(1, 'h')
    # and convert it back to a dataarray
    resample_hours = xarray.DataArray(resample_hours, coords={'resample_prd': da.resample_prd})
    seas_mean[name] = da * resample_hours
    # and do a single param fit too all the data..
    gev_fit_file = file.parent / 'fits' / ('gev_fit_' + file.name)
    stacked = da.stack(idx=['x', 'y', 'time']).dropna('idx')
    stacked = stacked.where(stacked > 0.5)
    seas_gev_fit[name] = gev_r.xarray_gev(stacked, dim='idx', file=gev_fit_file, recreate_fit=recreate_fit)
    my_logger.info(f'Loaded {name} and fit data')

    # get in the gev fit data
    for dct, base in zip([gev_fit, gev_fit_bs, gev_t_fit, gev_t_fit_bs],
                         ['gev_fit', 'gev_fit_bs', 'gev_fit_temp', 'gev_fit_temp_bs']
                         ):
        gev_file = file.parent / 'fits' / (base + '.nc')
        dct[name] = xarray.open_dataset(gev_file).mean('sample').load()

    my_logger.info(f'Loaded {name}')

## now to plot them
prd = '1h'
label = commonLib.plotLabel()
fig, ((ax_fit, ax_median), (ax_change_loc, ax_change_sc)) = plt.subplots(2, 2, figsize=(8, 8),
                                                                         clear=True, layout='constrained',
                                                                         num='Sydney_sens'
                                                                         )

for (name, parameters), col in zip(seas_mean.items(), ['blue', 'orange', 'green', 'red']):
    gev = scipy.stats.genextreme(
        *seas_gev_fit[name].Parameters.sel(parameter=['shape', 'location', 'scale'], resample_prd=prd)
        )
    radar_data = seas_mean[name].sel(resample_prd=prd).stack(idx=['x', 'y', 'time'])
    osm, osr = scipy.stats.probplot(radar_data, dist=gev, fit=False)
    ax_fit.plot(osm[0], osm[1], label=name, ms=2, marker='.', linestyle='None', color=col)

ax_fit.axline((5.0, 5.0), slope=1.0, color='black', linestyle='--')
#ax_fit.axhline(0.5 * float(resample_hours.sel(resample_prd=prd)), color='black', linestyle='--')
ax_fit.set_xlabel(f'GEV fit (mm) ')
ax_fit.set_ylabel('Radar (mm)')
ax_fit.set_xscale('log')
ax_fit.set_yscale('log')
ax_fit.set_title(f' Rx{prd} fit vs data')
ax_fit.legend(loc='upper left', fontsize='small', handletextpad=0.1, handlelength=0.5)
# plot the medians/90% of the data.
for (name, maxRain), col in zip(seas_mean.items(), ['blue', 'orange', 'green', 'red']):
    q = maxRain.sel(resample_prd='1h').quantile([0.5,0.9],['x', 'y'])
    q.isel(quantile=0).plot(ax=ax_median, label=name, color=col, marker='o',ms=7,linestyle='None')
    q.isel(quantile=1).plot(ax=ax_median, color=col, linestyle='None',marker='+',ms=10)
ax_median.legend()

## now to plot the gev ratios.
for (name, ds), col in zip(gev_t_fit.items(), ['blue', 'orange', 'green', 'red']):
    parameters=ds.Parameters
    for p, ax_change in zip(['location', 'scale'], [ax_change_loc, ax_change_sc]):

        dp = f'D{p}_Tanom'
        ds = parameters.sel(parameter=dp) / parameters.sel(parameter=p)
        ds_bs = gev_t_fit_bs[name].Parameters.sel(parameter=dp) / gev_t_fit_bs[name].Parameters.sel(parameter=p)
        q = ds_bs.quantile([0.1, 0.9], dim='bootstrap_sample')
        ax_change.fill_between(q.resample_prd, q.isel(quantile=0), q.isel(quantile=-1), color=col, alpha=0.2)
        ds_bs.median('bootstrap_sample').plot(ax=ax_change, color=col, linestyle='--')
        ds.plot(ax=ax_change, label=name, color=col, marker='o')
        ax_change.set_title(f'D {p}/dT')
        ax_change.set_xlabel('Resample period')
        ax_change.set_ylabel('Fractional change')
        for v in [0, 0.075, 0.15]:
            ax_change.axhline(v, color='k', linestyle='--')
        ax_change.set_ylim(-0.2, 0.5)
fig.suptitle('Sydney sensitivity studies')
# add labels
for ax in [ax_fit, ax_change_loc, ax_change_sc]:
    label.plot(ax)

fig.show()
commonLib.saveFig(fig)

# lets do some sens studies
def proc_events(file,threshold=0.5):
    drop_vars=['xpos', 'ypos','fraction', 'sample_resolution', 'height','Observed_temperature','time']
    radar_dataset = xarray.load_dataset(file, drop_variables=drop_vars)
    radar_dataset = radar_dataset.sel(resample_prd=['30min', '1h', '2h','4h'])
    # convert radar_dataset to accumulations.

    msk = (radar_dataset.max_value > threshold)
    radar_msk = radar_dataset.where(msk)
    mn_temp = radar_msk.ObsT.mean(['quantv', 'EventTime'])
    radar_msk['Tanom'] = radar_msk.ObsT - mn_temp  #
    return radar_msk

from process_submit.process_gev_fits import comp_radar_fit
radar_msk = proc_events(ausLib.data_dir / 'processed' / 'Sydney_rain_melbourne' /
                        'events_seas_mean_Sydney_rain_melbourne_DJF.nc')
fit_t, fit_t_bs = comp_radar_fit(radar_msk, cov=['Tanom'])
print('fit ratios ',ausLib.comp_ratios(fit_t.Parameters.mean('sample')).to_dataframe().unstack().round(2))
# drop post 2020 data
msk = (radar_msk.t.dt.year < 2020)

fit_t, fit_t_bs = comp_radar_fit(radar_msk.where(msk), cov=['Tanom'])
print('fit ratios 2020-',ausLib.comp_ratios(fit_t.Parameters.mean('sample')).to_dataframe().unstack().round(2))
# and only post 2010
msk = (radar_msk.t.dt.year >= 2010)
fit_t, fit_t_bs = comp_radar_fit(radar_msk.where(msk), cov=['Tanom'])
print('fit ratios 2010+',ausLib.comp_ratios(fit_t.Parameters.mean('sample')).to_dataframe().unstack().round(2))
# larger threshhold
msk = (radar_msk.max_value > 2)
fit_t, fit_t_bs = comp_radar_fit(radar_msk.where(msk), cov=['Tanom'])
print('fit ratios threshold = 2 ',ausLib.comp_ratios(fit_t.Parameters.mean('sample')).to_dataframe().unstack().round(2))
# drop roughly 1/2 the data
msk = (radar_msk.EventTime%2 == 0)
fit_t, fit_t_bs = comp_radar_fit(radar_msk.where(msk), cov=['Tanom'])
print('fit ratios 1/2 the data ',ausLib.comp_ratios(fit_t.Parameters.mean('sample')).to_dataframe().unstack().round(2))
# remove temps near the zero anomaly.
msk = (np.abs(radar_msk.Tanom) > 0.5)
fit_t, fit_t_bs = comp_radar_fit(radar_msk.where(msk), cov=['Tanom'])
print('fit ratios |Anom T|>0.5 ',ausLib.comp_ratios(fit_t.Parameters.mean('sample')).to_dataframe().unstack().round(2))