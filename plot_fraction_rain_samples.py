# plot the fraction of samples with rain for 2020.
# plot the 2021 fraction of rain samples.
# for  Sellick (46), Melbourne (id=2), West Takone (id=52), Sydney/Wollongong (id=3),  Brisbane (id=50) Cairns (id=19) ,Mornington Island (36),
import math
import pathlib
import cartopy.crs as ccrs
import xarray
import ausLib
import logging
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger("MAIN")
ausLib.init_log(log,level='INFO')

root_dir = pathlib.Path.home()/'data/aus_rain_analysis'
fractions=dict()
summer_maximum = dict()
summer_hr = dict()
for city in ['adelaide','melbourne','wtakone','canberra','sydney','newcastle','grafton','brisbane','gladstone','cairns','mornington']:
    file = root_dir/city/f'processed_2020_{city}.nc'
    ds= xarray.open_dataset(file)
    fract = ds.rain_count.sum('time')/ds.rain_samples.sum('time')
    fract = fract.load()
    fractions[city] = fract
    log.info(f"Computed fraction rain for {city}")
    L=ds.time.dt.season=='DJF'
    mx=ds.max_rain.sel(time=L).max('time').load()
    hr = ds.time_max_rain.sel(time=L).dt.hour.load()
    summer_maximum[city] = mx
    summer_hr[city]=hr

## now to plot the fraction data
nreg = len(fractions)
ncols = math.ceil(math.sqrt(nreg))
nrows = math.ceil(nreg/ncols)

fig,axis = plt.subplots(ncols=ncols,nrows=nrows,squeeze=False,
                    num='fraction_present',figsize=(11,7),clear=True,layout='constrained',
                    subplot_kw=dict(projection=ccrs.PlateCarree()))
levels=[0.005,0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.5]
for (name,fract),ax in zip(fractions.items(),axis.flatten()):
    rng = np.sqrt(fract.x.astype('float') ** 2 + fract.y.astype('float') ** 2) / 1e3
    rng = rng.assign_coords(longitde=fract.longitude,latitude=fract.latitude)
    cm=fract.plot(ax=ax,x='longitude',y='latitude',cmap='RdYlBu',
                     add_colorbar=False,levels=levels,transform=ccrs.PlateCarree())
    rng.plot.contour(levels=np.linspace(0, 150, 6),colors='black',linestyles='solid',ax=ax,
                     transform=ccrs.PlateCarree(),x='longitude',y='latitude')
    ax.coastlines()
    ax.set_title(name)
    g=ax.gridlines(draw_labels=True)
    g.top_labels = False
    g.right_labels = False

fig.colorbar(cm,ax=axis,orientation='horizontal',spacing='uniform',extend='both',pad=0.075,aspect=40,fraction=0.075)
fig.show()
fig.savefig('figures/fraction.png',dpi=300)

## plot the summer mx.
fig,axis = plt.subplots(ncols=ncols,nrows=nrows,squeeze=False,
                    num='summer_mx',figsize=(11,7),clear=True,layout='constrained',
                    subplot_kw=dict(projection=ccrs.PlateCarree()))
levels=[10,20,50,100,200]
for (name,mx),ax in zip(summer_maximum.items(),axis.flatten()):
    rng = np.sqrt(mx.x.astype('float') ** 2 + mx.y.astype('float') ** 2) / 1e3
    rng = rng.assign_coords(longitde=mx.longitude,latitude=mx.latitude)
    cm=mx.plot(ax=ax,x='longitude',y='latitude',cmap='RdYlBu',
                     add_colorbar=False,levels=levels,transform=ccrs.PlateCarree())
    rng.plot.contour(levels=np.linspace(0, 150, 6),colors='black',linestyles='solid',ax=ax,
                     transform=ccrs.PlateCarree(),x='longitude',y='latitude')
    ax.coastlines()
    ax.set_title(name)
    g=ax.gridlines(draw_labels=True)
    g.top_labels = False
    g.right_labels = False

fig.colorbar(cm,ax=axis,orientation='horizontal',spacing='uniform',extend='both',pad=0.075,aspect=40,fraction=0.075)
fig.show()
fig.savefig('figures/summer_max.png',dpi=300)

# dist of hour
## plot the summer mx.
fig,axis = plt.subplots(ncols=ncols,nrows=nrows,squeeze=False,
                    num='summer_hr',figsize=(11,7),clear=True,layout='constrained')

for (name,hr),ax in zip(summer_hr.items(),axis.flatten()):

    dist=hr.plot.hist(ax=ax,bins=np.arange(0,25))
    ax.set_title(name)



fig.show()
fig.savefig('figures/summer_hr.png',dpi=300)
