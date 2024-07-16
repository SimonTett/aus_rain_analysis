# code to do prob plot for GEV fit.
import ausLib
import xarray
import commonLib
import scipy.stats
import matplotlib.pyplot as plt
name='Sydney_rain_melbourne'
#name='Melbourne_rain_melbourne'
dir = ausLib.data_dir/'processed'/name
events = xarray.load_dataset(dir/f'events_seas_mean_{name}_DJF.nc').sel(resample_prd='1h') # events
fits= xarray.load_dataset(dir/'fits'/'gev_fit_temp.nc').sel(resample_prd='1h') # fits to data
params = fits.Parameters.mean('sample')
# normalise the data
anomT = events.ObsT - events.ObsT.mean('EventTime')
loc = params.sel(parameter='location')+anomT*params.sel(parameter='Dlocation_Tanom') # location param
scale = params.sel(parameter='scale')+anomT*params.sel(parameter='Dscale_Tanom') # scale param
L=events.max_value > 0.5
norm_data = (events.max_value.where(L)-loc)/scale
norm_data = norm_data.stack(idx=['quantv','EventTime']).dropna('idx')
gev= scipy.stats.genextreme( c=params.sel(parameter='shape'))
fig = plt.figure(num='prob_plot',clear=True)
ax = fig.add_subplot(111)
osm, osr = scipy.stats.probplot(norm_data,dist=gev,fit=False)
ax.scatter(osm,osr,marker='.',s=1)
ax.set_xlabel('Theoretical quantiles')
ax.set_ylabel('Ordered Values')
ax.axline((1,1),slope=1,color='r')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(0.1,None)
ax.set_xlim(0.1,None)
ax.plot(*[gev.median()]*2,'ro')
ax.set_title(name)
fig.show()