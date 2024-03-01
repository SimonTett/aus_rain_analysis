# plot radar avg values.
from datetime import datetime

import matplotlib.dates
import xarray
import matplotlib.pyplot as plt
import pathlib
import cartopy.crs as ccrs
import numpy as np

prds = dict(adelaide=slice('2000', None),
            brisbane=slice('2001', None),
            melbourne=slice('2003', None),
            sydney=slice("2000",None),
            wtakone=slice("2004",None)
            )
names = ["adelaide", "brisbane", "cairns", "canberra", "gladstone", "grafton", "melbourne",
         "mornington", "newcastle", "sydney", "wtakone"]
#names=['sydney']
names=['newcastle']
for name in names:
    file = f"../../data/aus_rain_analysis/radar/{name}/processed_{name}.nc"

    ds = xarray.open_dataset(file)
    # fix long/lat
    for c in ['longitude', 'latitude']:
        try:
            ds[c] = ds[c].isel(time=0).drop_vars('time')
        except ValueError:
            pass
    rng = np.sqrt(ds.x.astype(float) ** 2 + ds.y.astype(float) ** 2) / 1e3  # range in km
    rng = rng.assign_coords(longitude=ds['longitude'], latitude=ds['latitude'])
    med = ds.count_1h_thresh.resample(time='QS-DEC').mean().median(['x', 'y']).resample(time='MS').ffill()
    thresh = 2.5
    #msk = (ds.count_1h_thresh < 200) | ((ds.count_1h_thresh < (med * thresh)) & (ds.count_1h_thresh > med / thresh))
    msk = (ds.count_1h_thresh < (med * thresh)) & (ds.count_1h_thresh > med / thresh)
    ds_msk = ds.where(msk)
    ## plot the data.

    fig, axis = plt.subplots(nrows=2, ncols=3, num=f"{name}_max_mean", figsize=(11, 8),
                             layout='constrained', clear=True, subplot_kw=dict(projection=ccrs.PlateCarree()))
    fig_ts, axis_ts = plt.subplots(nrows=2, ncols=3, num=f"{name}_ts", figsize=(11, 8),
                                   layout='constrained', clear=True, sharex=True)
    al = True
    kw_colorbar = dict(orientation='horizontal', fraction=0.1, aspect=40, pad=0.05, spacing='uniform')
    cmap = 'RdYlBu'

    coord_dict = dict(x='longitude', y='latitude')
    for dd, axe, axe_ts, title in zip([ds, ds_msk], axis, axis_ts, ['Raw', 'Masked']):
        for ax, axt, v, t in zip(axe, axe_ts, ['mean_rain', 'max_rain', 'count_1h_thresh'],
                                 ['DJF rain (mm)', 'Max Rain (mm/h)', 'Rain hours']):
            ok_fract = (~dd[v].isnull()).mean(['x', 'y'])
            da = dd[v].where(ok_fract > (np.pi/4)*0.5)  # mask out times where too much missing data.(want 50 % of possible data
            resamp = da.resample(time='QS-DEC')  # summer.
            if v == 'mean_rain':
                da = resamp.mean()
                da *= 24 * 90  # convert to mm/season
            elif v == 'count_1h_thresh':
                da = resamp.sum(min_count=1)
            elif v == 'max_rain':
                da = resamp.max()
            else:
                da = resamp.mean()
            summer = da.time.dt.season == 'DJF'
            da = da.where(summer, drop=True)

            cbar_kwargs = kw_colorbar.copy()
            cbar_kwargs.update(label=t)
            sel = prds.get(name,slice(None,None))
            da.sel(time=sel).median('time').plot(robust=True, ax=ax, cmap=cmap, cbar_kwargs=cbar_kwargs, **coord_dict)
            ax.coastlines()
            rng.plot.contour(ax=ax, colors=['black'], linestyle='solid', levels=np.arange(0, 150, 30), **coord_dict)
            q = da.quantile([0.5, 0.8, 0.9, 0.99, 0.999], ['x', 'y'])  # .assign_coords(year=da.time.dt.year)
            q.plot.line(ax=axt, x='time', add_legend=al)
            for time_rng in [sel.start,sel.stop]:
                # Plot a horizontal line corresponding to the time range
                if time_rng:
                    start_time = matplotlib.dates.date2num(datetime.strptime(time_rng, '%Y'))
                    axt.axvline(start_time, color='r', linestyle='--')
            al = False  # no more legends
            axt.set_xlabel('Time')
            axt.set_ylabel(t)

    fig.show()
    fig_ts.show()
    fig.savefig(f'figures/radar_{name}_maps.png',dpi=300)
    fig_ts.savefig(f'figures/radar_{name}_ts.png',dpi=300)