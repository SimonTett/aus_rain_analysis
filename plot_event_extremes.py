# plot the rainfall event extremes vs their temperatures
# will show for 30min and 1 hour events. Different colors/symbols to be used.
import statsmodels.graphics.gofplots

import ausLib
import xarray
import pandas as pd
my_logger  = ausLib.setup_log(1)
## load in the data
rain_extremes=dict()
temp_extremes=dict()
time_extremes=dict()
calib='_rain_brisbane' # calibration
max_dbz = 55.0 # should be the events file.
for site in ausLib.site_numbers.keys():
    name = site+calib
    event_file = ausLib.data_dir/f'processed/{name}/events_{name}_DJF.nc'
    if not event_file.exists():
        raise FileNotFoundError(f'No event file for {site} with {event_file}')
    ds = xarray.load_dataset(event_file) # median extreme event
    resample_hours = pd.to_timedelta(ds.resample_prd)/pd.Timedelta(1,'h')
    # and convert it back to a dataarray
    resample_hours = xarray.DataArray(resample_hours, coords={'resample_prd':ds.resample_prd})
    max_accum = ds.max_value*resample_hours
    rain_extremes[site] = max_accum
    temp_extremes[site] = ds.ObsT
    time_extremes[site] = ds.t
    # compute the max rain possible.

    my_logger.info(f'Processed {site}')
# get the max rain -- pick Melbourne
name='Melbourne'+calib
event_file = ausLib.data_dir/f'processed/{name}/events_{name}_DJF.nc'

ds = xarray.open_dataset(event_file) # median extreme event
resample_hours = pd.to_timedelta(ds.resample_prd)/pd.Timedelta(1,'h')
# and convert it back to a dataarray
resample_hours = xarray.DataArray(resample_hours, coords={'resample_prd':ds.resample_prd})
mx_rain = ds.attrs['to_rain'][0]*(10 ** (max_dbz * ds.attrs['to_rain'][1] / 10.0))
mx_rain = resample_hours*mx_rain

## now to plot the data.
fig,axes = ausLib.std_fig_axs(f'event_extremes',sharey=True)
offset  = 0
for resamp_prd,color,marker in zip(['1h'],['red','blue','green'],['o','h','d']):
    sel=dict(resample_prd=resamp_prd,quantv=0.9)
    for site,ax in axes.items():
        ax.scatter(temp_extremes[site].sel(**sel)+offset,rain_extremes[site].sel(**sel),s=3,color=color,marker=marker)
        ax.set_title(site)
        #ax.set_xlabel('Temperature (C)')
        #ax.set_ylabel('Rainfall (mm)')
    offset += 0.1
fig.suptitle(f'Rainfall accum vs Temperature')
fig.show()
## plot the qq plots for the GEV fits at quantv = 0.1, 0.5 & 0.9
fig,axes = ausLib.std_fig_axs(f'event_qq')
prd='1h'
quantv=[0.1,0.5,0.9]
colors=['blue','orange','green','red']
import scipy.stats
for site,ax in axes.items():
    da = rain_extremes[site]
    print(da.max(['quantv','EventTime']))
    for col,prd in zip(colors,['30min','1h','2h','4h']):
        sel = dict(resample_prd=prd)
        data = da.sel(**sel).isel(quantv=-2)
        data = data.where(data > 0.5).dropna('EventTime')
    # statsmodels.graphics.gofplots.qqplot(ds.dropna('EventTime'),
    #                                      dist=scipy.stats.genextreme,fit=True,line='45',ax=ax,color='red',fmt='',ms=2)
        dist_p = scipy.stats.genextreme.fit(data)
        gev  = scipy.stats.genextreme(*dist_p)
        osm,osr = scipy.stats.probplot(data,dist=gev)
        ax.plot(osm[0],osm[1],color=col,label=prd,ms=4,marker='.',linestyle='-')
        # add on the max if in range.
        if mx_rain.sel(resample_prd=prd) < ax.get_ylim()[1]:
            ax.axhline(mx_rain.sel(resample_prd=prd),color=col,linestyle='--')


    ax.set_title(site)
    ax.axline((5.0,5.0),slope=1.0,color='black',linestyle='--')
    ax.set_xlabel('GEV fit (mm)')
    ax.set_ylabel('Radar (mm)')
    # ax.set_yscale('log')
    # ax.set_xscale('log')stre

handles, labels = axes['Melbourne'].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, loc=(0.4, 0.9), fontsize='small')
fig.suptitle(f'QQ plot')
fig.show()


## see how many TC dates we have.
import cartopy.geodesic
import numpy as np
g = cartopy.geodesic.Geodesic()
tracks=xarray.load_dataset(ausLib.common_data/'IBTrACS.since1980.v04r01.nc')

for site,no in ausLib.site_numbers.items():
    info = ausLib.site_info(no)
    dist=g.inverse(info.iloc[0].loc[['site_lon','site_lat']].astype('float').values,
                   np.array([tracks.lon,tracks.lat]).T)
    L = dist[:,0] < 500e3 # within 500 km
    breakpoint()
    L= L.reshape(tracks.lon.shape)
    coords = [tracks.lon.storm,tracks.lon.date_time]
    L=xarray.DataArray(L, coords=coords)

    # dates
    tc_dates=np.unique(np.datetime_as_string(tracks.time.values.flatten()[L],
                                             unit='D'))[:-1]

    # and see how many dates of max we have.
    match = time_extremes[site].dt.strftime('%Y-%m-%d').isin(tc_dates.tolist())
    no = match.sum().values
    if no == 0:
        print(f'{site}: No tc dates ')
    else:
        print(f'{site}: {no} tc dates')
        # print out the unique dates
        udates = np.unique(time_extremes[site].dt.strftime('%Y-%m-%d').values[match])
        print(site, udates)
        # and the names
        L2 = tracks[L].time.dt.strftime('%Y-%m-%d').isin(udates)
        print(site,np.unique(tracks.name[L2]))


