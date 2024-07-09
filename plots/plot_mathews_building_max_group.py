# Plot the max rain event at the Mathews building
import ausLib
import matplotlib.pyplot as plt
import xarray
import cartopy.crs as ccrs
import pandas as pd
import commonLib
file = ausLib.data_dir/'processed/Sydney_rain_melbourne/seas_mean_Sydney_rain_melbourne_DJF.nc'
ds=xarray.open_dataset(file)
coords=(151.23430027582538,-33.91743295047844)
proj = ausLib.radar_projection(ds.proj.attrs)
x,y = proj.transform_point(*coords, ccrs.PlateCarree())
mathews_rain=ds.max_rain_rate.sel(resample_prd='1h').sel(x=x, y=y, method='nearest')
mathews_time = ds.time_max_rain_rate.sel(resample_prd='1h').sel(x=x, y=y, method='nearest')
time_max_idx=int(mathews_rain.argmax())
time_of_max_rain=pd.to_datetime(mathews_time.isel(time=time_max_idx).values)
date=time_of_max_rain.strftime('%Y-%m-%d')
max_rain = ds.max_rain_rate.sel(resample_prd='1h').isel(time=time_max_idx)
max_rain_time = ds.time_max_rain_rate.sel(resample_prd='1h').isel(time=time_max_idx)
max_rain_group = max_rain.where(max_rain_time.dt.strftime('%Y-%m-%d')==date)


##now to make some plots
fig, axs = plt.subplots(nrows=1,ncols=2,subplot_kw=dict(projection=proj),clear=True,figsize=(12,6),
                       num='Mathews_max',sharex=True,sharey=True,layout='constrained')

levels=[1,5,10,20,40,50,75,100]
cmap='YlGnBu'
cbar_kwargs=dict(label='mm/h',orientation='horizontal',fraction=0.1,pad=0.05,aspect=40,extend='both')
cm=max_rain.plot(ax=axs[0],levels=levels,add_colorbar=False,cmap=cmap)
axs[0].set_title(f'Rx1h for {max_rain.time.dt.strftime("%Y-%m").values}')
max_rain_group.plot(ax=axs[1],levels=levels,add_colorbar=False,cmap=cmap)
axs[1].set_title(f'Rx1h group for {date}')

for ax in axs:
    ax.coastlines()
    ax.plot(*coords,marker='o',color='black',transform=ccrs.PlateCarree(),markersize=12,mew=1.5,alpha=0.5)
    ax.plot(0,0,marker='x',color='red',markersize=20,mew=2)

fig.colorbar(cm,ax=axs, **cbar_kwargs)
fig.show()
commonLib.saveFig(fig)




