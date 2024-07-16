# Plot the max rain event at the Mathews building
import numpy as np

import ausLib
import matplotlib.pyplot as plt
import xarray
import cartopy.crs as ccrs
import pandas as pd
import commonLib
file = ausLib.data_dir/'processed/Sydney_rain_melbourne/seas_mean_Sydney_rain_melbourne_DJF.nc'
ds=xarray.open_dataset(file).sel(resample_prd='1h')
coords=(151.23430027582538,-33.91743295047844) # Where mathews building is.
proj = ausLib.radar_projection(ds.proj.attrs)
x,y = proj.transform_point(*coords, ccrs.PlateCarree())
mathews_rain=ds.max_rain_rate.sel(x=x, y=y, method='nearest')
mathews_time = ds.time_max_rain_rate.sel(x=x, y=y, method='nearest')
time_max_idx=int(mathews_rain.argmax())
time_of_max_rain=pd.to_datetime(mathews_time.isel(time=time_max_idx).values)
# utc hour of 14:00 corresponds to roughly 00 in Eastern Australia.
ref_time = '1970-01-01T14:00'
group_fn = lambda time  : np.floor(((time - np.datetime64(ref_time)) / np.timedelta64(1, 'D')))
g_tgt = group_fn(time_of_max_rain)

max_rain = ds.max_rain_rate.isel(time=time_max_idx)

max_rain_time = ds.time_max_rain_rate.isel(time=time_max_idx)
g = group_fn(max_rain_time)
msk = (g==g_tgt)
max_rain_group = max_rain.where(msk)
max_rain_time_group = max_rain_time.where(msk)
time_range = [max_rain_time_group.fillna(np.datetime64('2050-01-01')).min(),
              max_rain_time_group.fillna(np.datetime64('1970-01-01')).max()]


##now to make some plots

fig = plt.figure('mathews_max',clear=True,figsize=(7,6),layout='constrained')
subfigs = fig.subfigures(2,1,height_ratios=[10,3]) # add on sub-figures.
label = commonLib.plotLabel()
yr=max_rain_time.time.dt.year.values
axs = subfigs[0].subplots(1,2,sharex=True,sharey=True,subplot_kw=dict(projection=proj))


levels=[1,5,10,20,40,50,75,100]
cmap='YlGnBu'
cbar_kwargs=dict(label='mm',orientation='horizontal',fraction=0.1,pad=0.05,aspect=40,extend='both')
cm=max_rain.plot(ax=axs[0],levels=levels,add_colorbar=False,cmap=cmap)
axs[0].set_title(f'Rx1h for {yr}/{yr-1999} DJF')
max_rain_group.plot(ax=axs[1],levels=levels,add_colorbar=False,cmap=cmap)
time_range_str = [t.dt.strftime('%Y-%m-%d:%H').values for t in time_range]
axs[1].set_title(f'Rx1h for {time_range_str[0]} - \n{time_range_str[1]}')

for ax in axs:
    ax.coastlines()
    ax.plot(*coords,marker='o',color='black',transform=ccrs.PlateCarree(),markersize=7,mew=1.5,alpha=0.5)
    ax.plot(0,0,marker='x',color='red',markersize=15,mew=5)
    label.plot(ax)

subfigs[0].colorbar(cm,ax=axs, **cbar_kwargs)


# and make a small plot showing the sorted values and the various quantiles..
mx = max_rain_group.stack(idx=['x','y']).dropna('idx').values
mh = (max_rain_time_group.stack(idx=['x','y']).dropna('idx').dt.hour.values+10)%24 # UTC 14 = 00 in Eastern Australia
order_values = mx.argsort()  # indices that sort the data
mx = mx[order_values]
mh = mh[order_values]

axs = subfigs[1].subplots(1,2)


axs[0].plot(mx,drawstyle='steps-pre')
axs[0].set_title('Ordered Rx1h')
axs[0].axhline(mathews_rain.max(),color='red',linestyle='--')
axs[1].plot(mh,marker='o',linestyle='None',ms=3)
axs[1].set_title('Hour of Max Rain')
axs[0].set_ylabel('Rx1h (mm)')
axs[1].set_ylabel('LSH')
ticks=np.arange(0,25,6)
labels=[str(tick) for tick in ticks]
axs[1].set_yticks(ticks,labels=labels,size='small')
qv=np.linspace(0.1,0.9,9)
labels = [f'{q:2.1f}' for q in qv]
for ax in axs:
    ax.set_xticks(qv*len(mx),labels=labels)
    ax.set_xlabel('Rx1h Quantile')
    label.plot(ax)


fig.show()
commonLib.saveFig(fig)





