# plot sensitivity studies for Sydney.
# Calibration -- melbourne and brisabne + melbourne calibration with largeer dbz range.

import ausLib
import pathlib
import xarray
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import commonLib
import numpy as np

my_logger = ausLib.setup_log(1)
event_files = list((ausLib.data_dir / 'processed').glob('Sydney_rain*/events*_Sydney_rain_*_DJF.nc'))
events={}
gev_t_fit=dict()
gev_t_fit_bs=dict()
rain_range=dict()
for file in event_files:
    name = file.parent.name.replace('Sydney_rain_','').capitalize()+ ' conv'
    ds=xarray.load_dataset(file)
    dbz= np.array(ds.attrs.get('dbz_range',[15,55]))
    da = xarray.open_dataset(file).max_value.sel(quantv=1).load()
    resample_hours = pd.to_timedelta(da.resample_prd) / pd.Timedelta(1, 'h')
    # and convert it back to a dataarray
    resample_hours = xarray.DataArray(resample_hours, coords={'resample_prd': da.resample_prd})
    events[name] = da*resample_hours
    # work out the rain range from the DBZ truncation.
    to_rain = ds.attrs['to_rain']
    rain_range[name] = [to_rain[0]*(10**(d*to_rain[1]/10)) for d in dbz]
    # get in the gev t fit data
    gev_t_file = file.parent / 'fits' / 'gev_fit_temp.nc'
    ds=xarray.open_dataset(gev_t_file).mean('sample')
    gev_t_fit[name] = ds.Parameters.load()
    # and the bootstrap.
    gev_t_bs_file = file.parent / 'fits' / 'gev_fit_temp_bs.nc'
    ds=xarray.open_dataset(gev_t_bs_file).mean('sample')
    gev_t_fit_bs[name] = ds.Parameters.load()

    my_logger.info(f'Loaded {name}')

## now to plot them
prd='30min'
label=commonLib.plotLabel()
fig, (ax_fit,ax_change_loc,ax_change_sc) = plt.subplots(1, 3, figsize=(8, 4),
                                     clear=True,layout='constrained',num='Sydney_sens')

for (name, parameters),col in zip(events.items(), ['blue', 'orange', 'green']):
    rn_lims = np.array(rain_range[name])*float(resample_hours.sel(resample_prd=prd))

    data = parameters.where(parameters > 0).sel(resample_prd=prd).dropna('EventTime')
    dist_p = scipy.stats.genextreme.fit(data)

    gev = scipy.stats.genextreme(*dist_p)
    osm, osr = scipy.stats.probplot(data, dist=gev)
    ax_fit.plot(osm[0], osm[1], label=name, ms=4, marker='.', linestyle='None',color=col)
    for r in rn_lims:
        ax_fit.axhline(r,  color=col, linestyle='--')
ax_fit.axline((5.0, 5.0), slope=1.0, color='black', linestyle='--')
ax_fit.axhline(0.5*float(resample_hours.sel(resample_prd=prd)), color='black', linestyle='--')
ax_fit.set_xlabel(f'GEV fit (mm) ')
ax_fit.set_ylabel('Radar (mm)')
ax_fit.set_xscale('log')
ax_fit.set_yscale('log')
ax_fit.set_title(f' Rx{prd} fit vs data')
ax_fit.legend(loc='upper left',fontsize='small',handletextpad=0.1,handlelength=0.5)
## now to plot the gev ratios.
for (name, parameters),col in zip(gev_t_fit.items(), ['blue', 'orange', 'green']):
    for p,ax_change in zip(['location', 'scale'],[ax_change_loc,ax_change_sc]):
        dp = f'D{p}_Tanom'
        ds = parameters.sel(parameter=dp) / parameters.sel(parameter=p)
        ds_bs = gev_t_fit_bs[name].sel(parameter=dp) / gev_t_fit_bs[name].sel(parameter=p)
        q=ds_bs.quantile([0.1,0.9],dim='bootstrap_sample')
        ax_change.fill_between(q.resample_prd, q.isel(quantile=0), q.isel(quantile=-1),color=col,alpha=0.2)
        ds_bs.mean('bootstrap_sample').plot(ax=ax_change, color=col, linestyle='--')
        ds.plot(ax=ax_change, label=name, color=col,marker='o')
        ax_change.set_title(f'D {p}/dT')
        ax_change.set_xlabel('Resample period')
        ax_change.set_ylabel('Fractional change')
        for v in [0,0.075,0.15]:
            ax_change.axhline(v,color='k',linestyle='--')
        ax_change.set_ylim(-0.2,0.5)
fig.suptitle('Sydney sensitivity studies')
# add labels
for ax in [ax_fit,ax_change_loc,ax_change_sc]:
    label.plot(ax)

fig.show()
commonLib.saveFig(fig)


