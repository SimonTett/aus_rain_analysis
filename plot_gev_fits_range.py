# fit radar maxes to a gev distribution with range & range^2 as covariates

import matplotlib.pyplot as plt
import scipy.stats

import ausLib
import xarray
import numpy as np
import commonLib
import pathlib
from R_python import gev_r


my_logger = ausLib.setup_log(1)
mx_data = dict()
gev_fits=dict()
gev_fits_rsqr=dict()
gev_fits_cube=dict()
gev_fits_recent= dict()
gev_fits_rsqr_recent = dict()
conversion = '_rain_melbourne'
long_radar_data = ausLib.read_radar_file("meta_data/long_radar_stns.csv")
sites = dict()
save_dir = ausLib.data_dir / 'processed/fits/fits_radius'
save_dir.mkdir(parents=True, exist_ok=True)
for site, id in ausLib.site_numbers.items():
    site_info = ausLib.site_info(id).iloc[-1]
    yr = site_info.postchange_start.year
    sites[site] = yr - 1

for site in sites.keys():
#for site in ['Melbourne','Adelaide']:
    name = site + conversion
    seas_file = ausLib.data_dir / f'processed/{name}/seas_mean_{name}_DJF.nc'
    if not seas_file.exists():
        raise FileNotFoundError(f'No season  file for {site} with {seas_file}')
    mx_rain = xarray.open_dataset(seas_file).max_rain_rate.load()
    # mask out small extremes
    mx_rain = mx_rain.where(mx_rain > 1)
    r = np.sqrt(mx_rain.x ** 2 + mx_rain.y ** 2) / 1000.
    mx_rain['range'] = r
    mx_data[site] = mx_rain
    my_logger.info(f'Extracted data for {site}')


# now do GEV fits
for site, mx_rain in mx_data.items():
    mx_rain_stack = mx_rain.stack(space_time=['x','y','time'])
    mx_rain_recent_stack = mx_rain.sel(time=slice(f'{sites[site]+1}-01-01',None)).stack(space_time=['x','y','time'])
    rng_sqr = (mx_rain_stack.range**2).rename('range_sqr')
    rng_cube = (mx_rain_stack.range**3).rename('range_cube')
    fit = gev_r.xarray_gev(mx_rain_stack, dim='space_time', file=save_dir/f'{site}_gev_fit.nc')
    fit_range_sqr = gev_r.xarray_gev(mx_rain_stack, cov = [mx_rain_stack.range,rng_sqr], dim='space_time',
                                     file=save_dir/f'{site}_gev_fit_range_sqr.nc')
    fit_range_cube = gev_r.xarray_gev(mx_rain_stack, cov = [mx_rain_stack.range,rng_sqr,rng_cube], dim='space_time',
                                     file=save_dir/f'{site}_gev_fit_range_cube.nc')
    gev_fits[site] = fit
    gev_fits_rsqr[site] = fit_range_sqr # store the fit with range and range^2 as covariates
    gev_fits_cube[site] = fit_range_cube
    # deal with recent data.
    rng_sqr = (mx_rain_recent_stack.range**2).rename('range_sqr')
    fit = gev_r.xarray_gev(mx_rain_recent_stack, dim='space_time',file=save_dir/f'{site}_gev_fit_recent.nc')
    fit_range_sqr = gev_r.xarray_gev(mx_rain_recent_stack, cov = [mx_rain_recent_stack.range,rng_sqr], dim='space_time',
                                        file=save_dir/f'{site}_gev_fit_range_sqr_recent.nc')
    gev_fits_recent[site] = fit
    gev_fits_rsqr_recent[site] = fit_range_sqr
    my_logger.info(f'Computed fits for {site}')

## now to plot
range = np.arange(10,stop=130,step=5)
range=xarray.DataArray(range,coords=[range],dims=['range'])
colors=['red','blue','green','purple']
fig, axs = ausLib.std_fig_axs('gev_fits_range',clear=True)
fig2,axs2 = ausLib.std_fig_axs('gev_fits_range_recent',clear=True)
for site in gev_fits_rsqr.keys():
    ax = axs[site]
    ax2 = axs2[site]
    fit = gev_fits[site]

    fit_rs = gev_fits_rsqr[site]
    fit_range_sqr=fit_rs.Parameters
    fit_range_sqr_recent = gev_fits_rsqr_recent[site].Parameters
    loc = (fit_range_sqr.sel(parameter='location')+
           range*fit_range_sqr.sel(parameter='Dlocation_range')+
           range**2*fit_range_sqr.sel(parameter='Dlocation_range_sqr'))
    scale = (fit_range_sqr.sel(parameter='scale')+
             range*fit_range_sqr.sel(parameter='Dscale_range')+
             range**2*fit_range_sqr.sel(parameter='Dscale_range_sqr'))

    loc_cube = (fit_range_sqr.sel(parameter='location')+
           range*fit_range_sqr.sel(parameter='Dlocation_range')+
           range**2*fit_range_sqr.sel(parameter='Dlocation_range_sqr')+
            range**3*fit_range_sqr.sel(parameter='Dlocation_range_cube'))
    scale_cube = (fit_range_sqr.sel(parameter='scale')+
             range*fit_range_sqr.sel(parameter='Dscale_range')+
             range**2*fit_range_sqr.sel(parameter='Dscale_range_sqr')+
              range**3*fit_range_sqr.sel(parameter='Dscale_range_cube'))

    loc_recent = (fit_range_sqr_recent.sel(parameter='location')+
           range*fit_range_sqr_recent.sel(parameter='Dlocation_range')+
           range**2*fit_range_sqr_recent.sel(parameter='Dlocation_range_sqr'))
    scale_recent = (fit_range_sqr_recent.sel(parameter='scale')+
             range*fit_range_sqr_recent.sel(parameter='Dscale_range')+
             range**2*fit_range_sqr_recent.sel(parameter='Dscale_range_sqr'))
    delta_aic = 1.0-float((fit_rs.AIC/fit.AIC).sel(resample_prd='1h'))
    delta_aic_recent = 1.0-float((gev_fits_rsqr_recent[site].AIC/gev_fits_recent[site].AIC).sel(resample_prd='1h'))
    for indx,col in enumerate(colors):

        loc.isel(resample_prd=indx).plot(x='range',ax=ax,label='location',add_legend=False,color=col)
        scale.isel(resample_prd=indx).plot(x='range',ax=ax,label='scale',linestyle='dashed',add_legend=False,color=col)

        loc_recent.isel(resample_prd=indx).plot(x='range',ax=ax2,label='location',add_legend=False,color=col)
        scale_recent.isel(resample_prd=indx).plot(x='range',ax=ax2,label='scale',linestyle='dashed',add_legend=False,color=col)
    #ax.set_title(fr'{site} $\Delta$AIC={np.round(delta_aic,-1):.0f}')
    #ax2.set_title(fr'{site} $\Delta$AIC={np.round(delta_aic_recent, -1):.0f}')

    ax.set_title(fr'{site} $\Delta$AIC={delta_aic:.2g}')
    ax2.set_title(fr'{site} $\Delta$AIC={delta_aic_recent:.2g}')
    #ax.legend()
fig.show()
fig2.show()
## plot median Rx1H and estimate median from fit.
fig, axs = ausLib.std_fig_axs('gev_fits_median',clear=True)
resample_prd='1h'
for site in gev_fits_rsqr.keys():
    ax = axs[site]

    fit_rs = gev_fits_cube[site]
    fit_range=fit_rs.Parameters.sel(resample_prd=resample_prd)
    data = mx_data[site].sel(resample_prd=resample_prd)
    data_median = data.groupby_bins('range',range).median().median('time')
    loc = (fit_range.sel(parameter='location')+
           range*fit_range.sel(parameter='Dlocation_range')+
           range**2*fit_range.sel(parameter='Dlocation_range_sqr')+
            range**3*fit_range.sel(parameter='Dlocation_range_cube')).squeeze()
    scale = (fit_range.sel(parameter='scale')+
                range*fit_range.sel(parameter='Dscale_range')+
                range**2*fit_range.sel(parameter='Dscale_range_sqr')+
                range**3*fit_range.sel(parameter='Dscale_range_cube')).squeeze()

    shape = (fit_range.sel(parameter='shape').broadcast_like(loc))
    gev = scipy.stats.genextreme(shape,loc=loc,scale=scale)
    gev_median = xarray.DataArray(gev.median(),dims=['range'],coords=dict(range=loc.coords['range']))
    gev_median.plot(ax=ax,label='fit')
    data_median.plot(ax=ax,label='data',marker='x')


    ax.set_title(f'{site}')
    ax.set_ylabel('Rx1H (mm)')
    ax.set_xlabel('Range (km)')

fig.show()



