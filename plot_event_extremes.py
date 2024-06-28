# plot the rainfall event extremes vs their temperatures
# will show for 30min and 1 hour events. Different colors/symbols to be used.
import ausLib
import xarray
import pandas as pd
my_logger  = ausLib.setup_log(1)
## load in the data
rain_extremes=dict()
temp_extremes=dict()
for site in ausLib.site_numbers.keys():
    name = f'{site}_rain_melbourne' # most sites calibrated using Melbourne rain
    if site in ['Melbourne','Brisbane','Wtakone']: # need to rerun these cases using Melbourne rain transfer fn
        name = f'{site}_rain'
    event_file = ausLib.data_dir/f'processed/{name}/events_{name}_DJF.nc'
    if not event_file.exists():
        raise FileNotFoundError(f'No event file for {site} with {event_file}')
    ds = xarray.load_dataset(event_file).sel(quantv=0.9) # median extreme event
    resample_hours = pd.to_timedelta(ds.resample_prd)/pd.Timedelta(1,'h')
    # and convert it back to a dataarray
    resample_hours = xarray.DataArray(resample_hours, coords={'resample_prd':ds.resample_prd})
    max_accum = ds.max_value*resample_hours
    rain_extremes[site] = max_accum
    temp_extremes[site] = ds.ObsT
    my_logger.info(f'Processed {site}')

## now to plot the data.
fig,axes = ausLib.std_fig_axs(f'event_extremes',sharey=True)
offset  = 0
for resamp_prd,color,marker in zip(['30min','1h','2h'],['red','blue','green'],['o','h','d']):
    sel=dict(resample_prd=resamp_prd)
    for site,ax in axes.items():
        ax.scatter(temp_extremes[site].sel(**sel)+offset,rain_extremes[site].sel(**sel),s=3,color=color,marker=marker)
        ax.set_title(site)
        #ax.set_xlabel('Temperature (C)')
        #ax.set_ylabel('Rainfall (mm)')
    offset += 0.1
fig.suptitle(f'Rainfall accum vs Temperature')
fig.show()

