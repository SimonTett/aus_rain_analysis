# plot Cairns, West Takone, Sydney & Brisbane pre and post radar mean rain as fn of range
import typing

import matplotlib.pyplot as plt
import ausLib
import xarray
import numpy as np
import commonLib
import pathlib
import cartopy
import cartopy.geodesic
import cartopy.crs as ccrs
def obs_precip_radius(obs_precip:xarray.DataArray,
                      lon:typing.Optional[float]=None,
                      lat:typing.Optional[float]=None,
                      radius:typing.Optional[np.ndarray] = None,
                      lon_coord:str = 'lon',
                      lat_coord:str = 'lat') -> xarray.DataArray:
    pts = np.meshgrid(obs_precip[lon_coord], obs_precip[lat_coord])
    pts = np.vstack([pts[0].flatten(), pts[1].flatten()]).T
    earth_geo = cartopy.geodesic.Geodesic()
    dist = earth_geo.inverse([lon, lat], pts)
    coords = [obs_precip.lon, obs_precip.lat]
    rad = xarray.DataArray(dist[:, 0].reshape(len(coords[0]), len(coords[1])) / 1e3,
                           coords=coords)
    result = obs_precip.groupby_bins(rad, radius).mean()
    return result
# use as follows:
# dd=ds_obs.precip.sel(lon=slice(site_rec.site_lon-2,site_rec.sat_lon+2),
#   lat=site_rec.site_lat,method='nearest').load()
# p_r = obs_precip_radius(dd,site_rec.site_lon,site_rec.site_lat,np.arange(0,200,50))
my_logger = ausLib.setup_log(1)
if not ('read_mean_rain_range' in locals() and read_mean_rain_range):
    mean_radar_rain = dict()
    mean_radar_rain_sens = dict()
    mean_gauge_rain=dict()
    conversion = '_rain_melbourne'
    sens_cons = '_rain_brisbane'
    long_radar_data = ausLib.read_radar_file("meta_data/long_radar_stns.csv")
    sites = dict()
    radius_bins=np.arange(0, 150, 25)
    obs_files = sorted(ausLib.agcd_rain_dir.glob('*.nc'))
    ds_gauge = xarray.open_mfdataset(obs_files).sel(time=slice('1997', None))
    for site, id in ausLib.site_numbers.items():
        site_info = ausLib.site_info(id).iloc[-1]
        yr = site_info.postchange_start.year
        sites[site] = yr - 1

    for site in sites.keys():
        name = site + conversion
        monthly_file = ausLib.data_dir / f'processed/{name}/monthly_mean_{name}.nc'
        if not monthly_file.exists():
            raise FileNotFoundError(f'No monthly   file for {site} with {monthly_file}')
        ds_rad = xarray.open_dataset(monthly_file)
        mn_rain = ds_rad.mean_raw_rain_rate.load() * ds_rad.time.dt.days_in_month * 24  # convert to mm
        mn_rain.attrs['units'] = 'mm'
        # get in sensitivity case
        name = site + sens_cons
        monthly_file = ausLib.data_dir / f'processed/{name}/monthly_mean_{name}.nc'
        if not monthly_file.exists():
            raise FileNotFoundError(f'No monthly   file for {site} with {monthly_file}')
        ds_rad_sens = xarray.open_dataset(monthly_file)
        mn_rain_sens = ds_rad_sens.mean_raw_rain_rate.load() * ds_rad.time.dt.days_in_month * 24  # convert to mm
        mn_rain_sens.attrs['units'] = 'mm'
        range = np.sqrt(mn_rain.x ** 2 + mn_rain.y ** 2) / 1000.
        mean_radar_rain[site] = mn_rain.groupby_bins(range, radius_bins).mean().load()
        mean_radar_rain_sens[site] = mn_rain_sens.groupby_bins(range, radius_bins).mean().load()
        # get inthe gauge data
        radar_proj = ausLib.radar_projection(ds_rad.proj.attrs)
        trans_coords = ccrs.PlateCarree().transform_points(radar_proj,
                                                           x=np.array([-125e3, 125e3]), y=np.array([-125e3, 125e3]))
        # get in the gauge data then inteprolate to radar grid and mask by radar data
        lon_slice = slice(trans_coords[0, 0], trans_coords[1, 0])
        lat_slice = slice(trans_coords[0, 1], trans_coords[1, 1])
        obs_rgn = dict(lon=lon_slice,  lat = lat_slice)
        longitude = mn_rain.longitude.isel(time=0).squeeze(drop=True)
        latitude = mn_rain.latitude.isel(time=0).squeeze(drop=True)
        time = mn_rain.time
        gauge = ds_gauge.precip.sel(**obs_rgn).interp(lon=longitude, lat=latitude, time=time).where(mn_rain.notnull())
        mean_gauge_rain[site] = gauge.groupby_bins(range, radius_bins).mean().load()
        my_logger.info(f'Loaded data for {site}')
    read_mean_rain_range  = True
    my_logger.info('Loaded all data')

def split_data(data,time):
    pre = data.sel(time=slice(None, time))
    post = data.sel(time=slice(time, None))
    pre_yrs = len(pre.time)
    if pre_yrs > 0:
        pre = pre.resample(time='YE').sum()
    else:
        pre = None

    post_yrs = len(post.time)
    if post_yrs > 0:
        post = post.resample(time='YE').sum()
    else:
        post = None
    return pre, post

fig,axs = ausLib.std_fig_axs('monthly_mean_rain_range',sharex=True,sharey=True,clear=True)
for (site, break_yr) in sites.items():
    ax = axs[site]
    pre, post  = split_data(mean_radar_rain[site], f'{break_yr}-12-31')
    pre_gauge, post_gauge = split_data(mean_gauge_rain[site], f'{break_yr}-12-31')
    pre_sens, post_sens = split_data(mean_radar_rain_sens[site], f'{break_yr}-12-31')
    if pre is not None:
        #pre.plot.line(x='group_bins',ax=ax,linestyle='-',add_legend=False,label='_None',alpha=0.5)
        pre_yrs = len(pre.time)
        sd = pre.std('time') / np.sqrt(pre_yrs)
        mn = pre.mean('time')
        x = [v.mid for v in pre.group_bins.values]
        ax.fill_between(x, mn - 2 * sd, mn + 2 * sd, alpha=0.5, color='red')
        mn.plot(x='group_bins', ax=ax,  label=f"Pre Radar", color='red', linewidth=3)
        pre_gauge.mean('time').plot(x='group_bins', ax=ax,  label=f"Pre gauge", color='maroon', linewidth=3)
        pre_sens.mean('time').plot(x='group_bins', ax=ax, linestyle='dashed', label=f"Pre Sens", color='red', linewidth=3)
    if post is not None:
        post_yrs = len(post.time)
        sd = post.std('time') / np.sqrt(post_yrs)
        mn = post.mean('time')
        mn.plot(x='group_bins', ax=ax,  label=f"Post Radar", color='skyblue', linewidth=3)
        x = [v.mid for v in post.group_bins.values]
        ax.fill_between(x, (mn - 2 * sd).values, (mn + 2 * sd).values, alpha=0.5, color='skyblue')
        post_gauge.mean('time').plot(x='group_bins', ax=ax,  label=f"Post gauge", color='blue', linewidth=3)
        post_sens.mean('time').plot(x='group_bins', ax=ax, linestyle='dashed', label=f"Post Sens", color='skyblue', linewidth=3)
    ax.set_title(site)
    #ax.set_yscale('log')
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Rainfall (mm)')
    ax.set_yscale('log')
    #ax.legend()
handles, labels = axs['Melbourne'].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2,fontsize='small',loc=(0.4, 0.9),handletextpad=0.2,handlelength=3.,columnspacing=1.0)
fig.suptitle('Ann Total rain')
fig.show()
commonLib.saveFig(fig, savedir=pathlib.Path('extra_figs'))
commonLib.saveFig(fig, savedir=pathlib.Path('extra_figs'), transpose=True)

"""# lets do all the Sydney sens studies.
files = list((ausLib.data_dir / 'processed').glob('Sydney_*/seas_mean_Sydney*_DJF.nc'))
fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True, clear=True, num='Sydney_sens_mean_range')
break_year = sites['Sydney']
for file, ax in zip(files, axs.flatten()):

    mn_rain = xarray.open_dataset(file).mean_raw_rain_rate.load() * 91 * 24  # convert to mm
    mn_rain.attrs['units'] = 'mm'
    r = np.sqrt(mn_rain.x ** 2 + mn_rain.y ** 2) / 1000.
    mn_rain['range'] = r
    mn_rain = mn_rain.groupby_bins('range', np.arange(0, 150, 10)).mean().load()
    pre = mn_rain.sel(time=slice(None, f'{break_year}'))
    post = mn_rain.sel(time=slice(f'{break_year + 1}', None))
    pre_yrs = len(pre.time)
    post_yrs = len(post.time)
    if pre_yrs > 0:
        sd = pre.std('time') / np.sqrt(pre_yrs)
        mn = pre.mean('time')
        x = [v.mid for v in pre.group_bins.values]
        mn.plot(x='group_bins', ax=ax, linestyle='-', label=f"Pre {break_year} N={pre_yrs}", color='red', linewidth=3)
        ax.fill_between(x, mn - 2 * sd, mn + 2 * sd, alpha=0.5, color='red')

    if post_yrs > 0:
        sd = post.std('time') / np.sqrt(post_yrs)
        mn = post.mean('time')
        mn.plot(x='group_bins', ax=ax, linestyle='--', label=f"Post {break_year} N={post_yrs}", color='k', linewidth=3)
        x = [v.mid for v in post.group_bins.values]
        ax.fill_between(x, (mn - 2 * sd).values, (mn + 2 * sd).values, alpha=0.5, color='k')

    ax.set_title(file.parent.name)
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Rainfall (mm)')
    ax.legend()
    ax.set_ylim(0, 300)
fig.suptitle('Sydney sens studies: Total DJF rain')

commonLib.saveFig(fig, savedir=pathlib.Path('extra_figs'))
"""