# Plot the spatial median of the maximum seasonal rain_rate and spatial mean of  mean rain.
# will do for both seasonal and monthly data.
import xarray
import matplotlib.pyplot as plt
import matplotlib.units
import ausLib
import commonLib
import numpy as np
site='Sydney'
name='Sydney_rain_melbourne'
resample_prd='1h'
seas_data = f'/home/z3542688/data/aus_rain_analysis/radar/processed/{name}/seas_mean_{name}_DJF.nc'
ds=xarray.open_dataset(seas_data)
ds_mon = xarray.open_mfdataset(f'/home/z3542688/data/aus_rain_analysis/radar/summary/{name}/*_rain.nc',combine='by_coords').sortby('time')
# get rid of where we don't have any data < 1000 samples.
L = (ds_mon.count_raw_rain_rate > 1000).load()
ds_mon = ds_mon.where(L,drop=True)
radar_info=ausLib.site_info(ausLib.site_numbers[site])
fig,axs = plt.subplots(nrows=2,ncols=1,num=f'{name}_max_mean',figsize=(7,9),layout='constrained',clear=True,sharex=True)
(ax_med,ax_mn)=axs
style=dict(linestyle='None',marker='o')
ds_mon.sel(resample_prd=resample_prd).max_rain_rate.median(['x','y']).plot(ax=ax_med,color='red',label='Monthly',**style)
ds.sel(resample_prd=resample_prd).max_rain_rate.median(['x','y']).plot(ax=ax_med,color='blue',label='DJF',**style)
ds_mon.mean_raw_rain_rate.mean(['x','y']).plot(ax=ax_mn,color='red',label='Monthly',**style)
ds.mean_raw_rain_rate.mean(['x','y']).plot(ax=ax_mn,color='blue',label='DJF',**style)
ax_med.set_ylabel('rain rate (mm/h)')
ax_mn.set_ylabel('rain rate (mm/h)')
ax_med.set_title(f'{site} Median Max')
ax_mn.set_title(f'{site} Mean')
# plot potential breakpoints.
for ax in axs:
    loc = ax.get_ylim()
    loc=np.mean(loc)
    for name, row in radar_info.iterrows():
        try:
            ax.axvline(row['prechange_end'],color='blue',linestyle='dashed')
        except matplotlib.units.ConversionError:
            pass
        ax.text(row['postchange_start'],loc,row['radar_type'])
        ax.axvline(row['postchange_start'],color='green',linestyle='dashed')
    ax.legend()
fig.show()
commonLib.saveFig(fig)
