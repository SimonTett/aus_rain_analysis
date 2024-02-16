# plot the radar data from three processed sites.
# step 1 get them in!
import pathlib

import numpy as np
import cartopy.crs as ccrs

import ausLib
import xarray
import matplotlib.pyplot as plt

base_dir = pathlib.Path.home() / "data/aus_rain_analysis/radar/"
data_set = dict()
for name in ['grafton', 'sydney', 'port_hedland']:
    file = (base_dir / name) / f"processed_radar_{name}.nc"
    data_set[name] = xarray.open_dataset(file)

## plot mean rain and where I can see any jumps.
mean = {k: (ds['mean_rain'].sel(time=slice('2010', None)).mean('time', keep_attrs=True) * 24.).load() for k, ds in
        data_set.items()}
minr={k:v['mean_rain'].sel(time=slice('2010', None)).min(['time'])*24 for k,v in data_set.items()}
maxr={k:v['mean_rain'].sel(time=slice('2010', None)).max(['time'])*24 for k,v in data_set.items()}
for k in mean.keys():
    mean[k].attrs.update(data_set[k].attrs)
# compute mean as mm.day
levels = np.linspace(0, 10, 11)
fig, axis = plt.subplots(nrows=2, ncols=2, num='radar_means', figsize=(12, 8), clear=True,
                         subplot_kw=dict(projection=ccrs.PlateCarree()))
axis_aus = axis[1][1]
for ax, radar in zip(axis.flatten(), mean.values()):
    cm = radar.plot(levels=levels, x='longitude', y='latitude', ax=ax, add_colorbar=False)
    delta = xarray.where(np.abs(radar - radar.shift(x=1, y=1)) > 0.2, 1.0, np.nan)
    delta.plot(levels=[1.0], x='longitude', y='latitude', ax=ax, add_colorbar=False, alpha=0.25)
    # delta.plot.contour(levels=[0.2], linewidths=[2], colors=['black'],
    #                   linestyles=['solid'], x='longitude', y='latitude', ax=ax)
    ax.coastlines()
    name = radar.attrs['instrument_name']
    ax.set_title(name)
    coords = [radar.attrs['origin_longitude'],
              radar.attrs['origin_latitude']]
    axis_aus.plot(coords[0], coords[1], marker='x', color='black')
    axis_aus.text(*coords, name, va='bottom', ha='center')

axis_aus.set_extent([110, 160, -45, -10])
axis_aus.coastlines()

fig.colorbar(cm, ax=axis, orientation='horizontal', fraction=0.05, pad=0.05, label='mm/day')
fig.show()
fig.savefig("figures/radar_means.png")

## plot timeseries of mean & median rain.
mean_ts = {k: (ds['mean_rain'].mean(['x', 'y'], keep_attrs=True) * 24.).load() for k, ds in data_set.items()}
median_ts = {k: (ds['mean_rain'].median(['x', 'y'], keep_attrs=True) * 24.).load() for k, ds in data_set.items()}
fig_ts, axis = plt.subplots(nrows=3, ncols=1, num='radar_ts', figsize=(12, 8), clear=True,
                            sharex=True,layout='constrained')

for ax,name in zip(axis,mean.keys()):
    mn=mean_ts[name]
    med = median_ts[name]
    mn.plot(ax=ax,color='green',linewidth=2,label='mean')
    med.plot(ax=ax,color='blue',linewidth=2,label='median')
    ax.set_title(name)
    ax.set_ylabel("Avg rain (mm/day)")
    ax.set_xlabel("Time")
ax.legend()
fig_ts.show()
fig_ts.savefig("figures/radar_avg_ts.png")
