# compute the ratio between the radial and circumferential components of time seps of extremes.
# doing this to test if absorption by extremes is causing problems
# If absorption is happening then expect day seperation changes in the radial direction to be larger than in angular direction.
# No real evidence of this in the data. (maybe out > 100 ish km)

# A bit of a  'mare investigating the data. There are issues with Brisbane data. I think the best approach
# is to filter out places where the monthly mean_rain_rate < 0.2 * mean_rain_rate.median(['x','y'])
# possibly also filter out places where the mean_rain_rate > 5 * mean_rain_rate.mean(['x','y'])
# do that as part of the seasonal aggregation when masking is done.

import pathlib
import time
import typing

import numpy as np
import xarray
import ausLib
import  matplotlib.pyplot as plt
import scipy.stats

import commonLib
from matplotlib.colors import LogNorm
import seaborn as sns

from plot_radar_gauge_ratio_range import linestyle


def angle_radius(x:xarray.DataArray,y:xarray.DataArray) -> xarray.Dataset:
    """
    compute the angle from the x-axis to the point (x,y) in radians
    :param x:
    :param y:
    :return:
    """
    angle = np.arctan2(y,x)
    radius = np.sqrt(y**2 + x**2)
    return xarray.Dataset(dict(angle=angle,radius=radius))

def interp_angular(ds:xarray.Dataset,) -> xarray.Dataset:
    """
    Interpolate the data in the angular direction
    :param ds:
    :return:
    """
    # Interpolate the data in the angular direction
    coords = angle_radius(ds.x,ds.y)
    ds = ds.assign_coords(coords).interp(angle=np.linspace(-np.pi,np.pi,181),method='linear')
    # Interpolate the data in the radial direction
    ds = ds.interp(radius=np.arange(0,120,2.),method='linear')
    breakpoint()
    return ds

def rotate(xcpt:xarray.DataArray,
           ycpt:xarray.DataArray) -> xarray.Dataset:
    """
    Rotate the points (xcpt,ycpt) by angle
    :param xcpt:
    :param ycpt:
    :return:
    """
    polar = angle_radius(xcpt.x,xcpt.y)
    cos = np.cos(polar.angle)
    sin = np.sin(polar.angle)
    angular = xcpt*cos - ycpt*sin
    radial = xcpt*sin + ycpt*cos
    ds = xarray.Dataset(dict(angular=angular,radial=radial))
    # Preserve existing coordinates
    for coord in xcpt.coords:
        if coord not in ds.coords:
            ds = ds.assign_coords({coord: xcpt.coords[coord]})
    # Assign the new coordinates to the dataset
    ds = ds.assign_coords(angle=polar.angle, radius=polar.radius)
    # Set the new coordinates as the main coordinates
    ds = ds.set_coords(['angle', 'radius'])

    return ds


def read_process(sites:list[str],
                 process:typing.Optional[typing.Callable] = None,
                 conversion:str='_rain_melbourne',
                 seas_file_fn:typing.Optional[typing.Callable] = None,
                 save_dir:typing.Optional[pathlib.Path] = None) -> dict :
    """
    Read in the data for a site and process it with the function process
    :param sites: List of sites to process
    :param conversion: Conversion used. name is site + conversion
    :param seas_file_fn: Function to get the file name for the season data. If None then use default which is
        ausLib.data_dir / f'processed/{name}/seas_mean_{name}_DJF.nc'
    :param process:Function to process the data. If None then do not process the data and just return it
    :param save_dir: Where to save the data. If None then do not save the data.
      Files, within this directory, are called {name}_processed.nc.
    :return: a dict of stuff indexed by names.

    Args:
        save_dir: Where to save data (if provided)
    """
    result=dict()
    if seas_file_fn is None: # no function provided so use default
        seas_file_fn = lambda name: ausLib.data_dir / f'processed/{name}/seas_mean_{name}_DJF.nc'

    for site in sites:
        name = site + conversion
        seas_file = seas_file_fn(name)
        if not seas_file.exists():
            raise FileNotFoundError(f'No season  file for {site} with {seas_file}')
        ds = xarray.load_dataset(seas_file)
        if process is None:
            result[name] = ds
        else:
            result[name]=process(ds)
    if save_dir is not None:
        save_dir.mkdir(exist_ok=True,parents=True)
        for name in result.keys():
            result[name].to_netcdf(save_dir/f'{name}_processed.nc')
    return result

def process_dtime(ds:xarray.Dataset) -> xarray.Dataset:
    """
    Process the dataset to get the time separation
    :param ds:
    :return:
    """
    dx = ds.time_max_rain_rate.shift(x=1)-ds.time_max_rain_rate.shift(x=-1)
    #dx = dx.mean('time')

    dy = ds.time_max_rain_rate.shift(y=1)-ds.time_max_rain_rate.shift(y=-1)
    #dy = dy.mean('time')
    result = rotate(dx,dy)
    result['radius'] = result['radius']/1000.
    return result

from scipy.interpolate import NearestNDInterpolator
def interp_NN_angular(ds: xarray.DataArray, specified_angle: np.ndarray, specified_radius: np.ndarray) -> xarray.DataArray:
    """
    Interpolate the data in the angular and radial directions using nearest points.
    :param ds: xarray.DataArray
    :param specified_angle: 1D array of angle values
    :param specified_radius: 1D array of radius values
    :return: xarray.DataArray
    """
    # Extract x and y coordinates
    x_coord='x'
    y_coord='y'
    x = ds.coords[x_coord]
    y = ds.coords[y_coord]

    # Compute angle and radius
    coords = angle_radius(x, y)
    angle = coords.angle.values.flatten()
    radius = coords.radius.values.flatten()
    # Create a meshgrid for the specified angle and radius
    angle_grid, radius_grid = np.meshgrid(specified_angle, specified_radius)
    # will need to iterate over the other coordinates so work them out.
    other_dims= [dim for dim in ds.dims if dim not in [x_coord,y_coord]]
    ds2= ds.stack(other_dims=other_dims)
    interp_coords = list(zip(angle, radius))
    result = []
    for idx in range(0,len(ds2.coords['other_dims'])):
        # Create the interpolator
        interpolator = NearestNDInterpolator(interp_coords, ds2.isel(other_dims=idx).values.flatten(),rescale=True)
        # rescale is essential. Without it nearest neighbour lookup will look only in the angle direction!
        interp = interpolator(angle_grid, radius_grid)

        interp = xarray.DataArray(interp, coords=[specified_radius, specified_angle],
                                  dims=['radius', 'angle'])
        interp = interp.expand_dims(other_dims=1,axis=-1)
        result.append(interp)
    result = xarray.concat(result,dim='other_dims').assign_coords(other_dims=ds2['other_dims'])

    result = result.unstack()
    return result

def process_dtime_interp(ds:xarray.Dataset) -> xarray.Dataset:
    """
    Process the dataset to get the time separation
    :param ds:
    :return:
    """
    # interpolate to angular coordinates
    dataArray = ds.time_max_rain_rate - ds.time
    dataArray = dataArray.dt.total_seconds()/3600. # difference in hours.

    da = interp_NN_angular(dataArray, np.linspace(-np.pi,np.pi,181), np.arange(0,150e3,5E3))
    dtime_r = da.shift(radius=1)-da.shift(radius=-1)
    dr = da.radius.shift(radius=1)-da.radius.shift(radius=-1)
    dtime_dr = dtime_r/dr

    dtime_phi = da.shift(angle=1)-da.shift(angle=-1)
    dphi = (da.angle.shift(angle=1)-da.angle.shift(angle=-1))*da.radius
    dtime_phi = dtime_phi/dphi

    result= xarray.Dataset(dict(dtime_dr=dtime_dr,dtime_dphi=dtime_phi,time=da))
    return result



conversion = '_rain_melbourne'
long_radar_data = ausLib.read_radar_file("meta_data/long_radar_stns.csv")

sites = list(ausLib.site_numbers.keys())
dtimes = read_process(sites,process=process_dtime_interp,conversion=conversion)

def comp_fract(ds:xarray.Dataset) -> xarray.Dataset:
    """
    Compute the fraction of the maximum rain rate
    :param ds:
    :return:
    """
    # interpolate to angular coordinates
    result = ds[['max_rain_rate','mean_rain_rate']]
    result['max_rain_rate'] = ds.max_rain_rate.where(ds.max_rain_rate > 0.5)

    result['total_rain'] = ds.mean_rain_rate * 90 * 24 # convert to mm from mm/hour
    result['total_raw_rain'] = ds.mean_raw_rain_rate*90*24
    result['fract_max'] = ds.max_rain_rate / result.total_rain
    return result

mn_max_rain = read_process(sites,process=comp_fract,conversion=conversion)


def process_gauge(ds:xarray.Dataset) -> xarray.Dataset:
    """
    Process the dataset to get the time separation
    :param ds:
    :return:
    """
    # interpolate to angular coordinates
    dataArray = ds.precip
    dataArray = dataArray.resample(time='QS-DEC').mean()
    L = dataArray.time.dt.season == 'DJF'
    dataArray = dataArray.where(L,drop=True)
    return dataArray

# need to read in the "observed" mean rain and plot ratios.
gauge_file_fn = lambda site: ausLib.data_dir / f'processed/{site}/gauge_rain_{site.replace('_rain','')}.nc'
gauge_rain = read_process(sites,process=process_gauge,conversion=conversion,
                          seas_file_fn=gauge_file_fn)
## plot

resample_prd='1h'
# Want to plot mx_rain/mn_rain and gauge_rain*mn_rain.mean()/mn_rain.mean()
fig, axs = ausLib.std_fig_axs('mx_rain',clear=True,sharex=True,sharey=True,add_projection=True)
fig2, axs2 = ausLib.std_fig_axs('mn_rain',clear=True,sharex=True,sharey=True,add_projection=True)
fig_dist,axs_dist = ausLib.std_fig_axs('kde_dist_ratio',figsize=(8,6),
                                   clear=True,layout='constrained')


levels_mx_mn = np.linspace(0,0.2,11)
levels_radar_gauge = np.round(np.geomspace(0.25,4,9),2) # log spacing (every sqrt(2)

cmap = 'RdYlBu'
for name,ds in mn_max_rain.items():
    total_rain_resample = ds.total_rain.sel(resample_prd=resample_prd)

    mx_r = ds.max_rain_rate.sel(resample_prd=resample_prd)
    L=mx_r > 0.5
    mx_r = mx_r.where(L)

    site,calib = name.split('_',1)
    ax = axs[site]
    ax2 = axs2[site]
    ax_dist= axs_dist[site]
    gauge_rain_r = gauge_rain[name].transpose()
    total_raw_rain = ds.total_raw_rain
    ratio = (total_raw_rain * gauge_rain_r.mean(['x', 'y']) / (gauge_rain_r * total_raw_rain.mean(['x', 'y']))).mean('time')

    fract_max = ds.fract_max.sel(resample_prd=resample_prd).mean('time')

    cm1=fract_max.plot(ax=ax,robust=True,cmap=cmap,levels=levels_mx_mn,add_colorbar=False)
    cm2=ratio.plot(ax=ax2,robust=True,cmap=cmap,
                   norm=LogNorm(vmin=levels_radar_gauge[0],vmax=levels_radar_gauge[-1]),
                   levels=levels_radar_gauge,add_colorbar=False)
    # plot the distribution of rain relative to median
    for v,colour,label in zip([total_raw_rain,gauge_rain_r],['green','blue'],['Radar','Gauge']):
        ratio = v/v.median(['x','y'])
        ratio = ratio.stack(dim=[...]).dropna('dim')
        res = scipy.stats.ecdf(ratio.values.flatten())
        res.cdf.plot(ax_dist,color=colour,label=label)
        # plot the mean distribution
        v2=v.mean('time')
        v2 /= v2.median(['x','y'])
        v2 = v2.stack(dim=[...]).dropna('dim')
        res = scipy.stats.ecdf(v2.values.flatten())
        res.cdf.plot(ax_dist,color=colour,linewidth=2)

    ax_dist.axvline(0.3,linestyle='--',color='black',linewidth=2)
    ax_dist.set_yscale('log')
    ax_dist.set_ylim(1e-4,1e1)
    ax_dist.set_xlim(0,2.0)
    ax_dist.set_title(f'{site} KDE dist')
    ax_dist.legend()

    # Add circles with radia from 0 to 150 km in steps of 25 km
    for a in [ax,ax2]:
        for radius in range(0, 151, 25):
            circle = plt.Circle((0, 0), radius*1e3, color='black', fill=False, linestyle='--')
            a.add_artist(circle)
        a.coastlines(color='purple', linewidth=1)
        a.plot(0.0,0.0,marker='h',color='red',markersize=10)
        a.set_title(site)
        a.set_xlabel('X (km)')
        a.set_ylabel('Y (km)')
        a.label_outer()
for  f,cm,label,level in zip([fig,fig2],[cm1,cm2],
                             ['Fraction Max/Total','Normalised Mean Radar/Gauge Rain'],
                             [levels_mx_mn,levels_radar_gauge]):
    cbar = f.colorbar(cm,cax=None,ax=f.axes,location='bottom',label=label,
             extend='both',pad=0.05,aspect=40)
    # from https://stackoverflow.com/questions/72772061/how-to-set-the-ticks-in-matplotlib-colorbar-to-be-at-the-minimum-and-maximum-val#72772248
    cbar.set_ticks(level)
    cbar.set_ticklabels([f'{l:.2f}' for l in level])
    f.suptitle(label)
    f.show()
    commonLib.saveFig(f,savedir=pathlib.Path('extra_figs'))
#ax3.legend(ncols=3)
#fig3.show()
commonLib.saveFig(fig_dist,savedir=pathlib.Path('extra_figs'))
## plot Grafton and Brisbane time series.
low_pts = dict(Grafton=dict(x=-50e3,y=97e3),Brisbane=dict(x=-30e3,y=-10e3))
names=[['Brisbane','Grafton']]
fig_ts,axs_ts = ausLib.std_fig_axs('Grafton_Brisbane_ts',mosaic=names,figsize=(8,6),
                                   clear=True,layout='constrained')
fig_map,axs_map = ausLib.std_fig_axs('Grafton_Brisbane_map',mosaic=names,figsize=(8,6),
                                      clear=True,layout='constrained',add_projection=True)

for site in axs_ts.keys():
    ax_map = axs_map[site]
    ax_ts=axs_ts[site]
    locn = low_pts[site]
    name=site+'_rain_melbourne'
    ds = mn_max_rain[name]
    total_raw_rain = ds.total_raw_rain
    gauge_rain_r = gauge_rain[name].transpose()
    ratio = (total_raw_rain * gauge_rain_r.mean(['x', 'y']) / (gauge_rain_r * total_raw_rain.mean(['x', 'y']))).mean('time')
    cm_ratio = ratio.plot(ax=ax_map,robust=True,cmap=cmap,
                   norm=LogNorm(vmin=levels_radar_gauge[0],vmax=levels_radar_gauge[-1]),
                   levels=levels_radar_gauge,add_colorbar=False)
    med = ds.total_raw_rain.median(['x','y'])
    low = ds.total_raw_rain.sel(**locn,method='nearest')
    gauge_low = gauge_rain_r.sel(**locn,method='nearest')
    gauge_med = gauge_rain_r.median(['x','y'])
    low.plot(ax=ax_ts,label='low',color='red',ds='steps-pre')
    gauge_low.plot(ax=ax_ts,label='gauge low',color='red',linestyle='--',ds='steps-pre')
    med.plot(ax=ax_ts,label='median',color='black',ds='steps-pre')
    gauge_med.plot(ax=ax_ts,label='gauge median',color='black',linestyle='--',ds='steps-pre')

    ausLib.plot_radar_change(ax_ts, ausLib.site_info(site), trmm=False)
    # Create a secondary y-axis
    ax_ratio = ax_ts.twinx()

    # Compute the ratios
    radar_ratio = med / low
    gauge_ratio = gauge_med / gauge_low

    # Plot the ratios on the secondary axis
    radar_ratio.plot(ax=ax_ratio, label='Radar Ratio', color='green', linestyle='solid', ds='steps-pre',linewidth=4,alpha=0.5)
    gauge_ratio.plot(ax=ax_ratio, label='Gauge Ratio', color='green', linestyle='--', ds='steps-pre',linewidth=4,alpha=0.5)

    # Customize the secondary axis
    ax_ratio.set_ylabel('Median/Low Ratio')
    ax_ratio.legend(loc='upper right')
    ax_ratio.set_title('')
    #ax_ratio.set_yscale('log')

    ax_ts.legend()
    ax_ts.set_title(site)
    ax_ts.set_yscale('log')
    for radius in range(0, 151, 25):
        circle = plt.Circle((0, 0), radius * 1e3, color='black', fill=False, linestyle='--')
        ax_map.add_artist(circle)
    ax_map.coastlines(color='purple', linewidth=1)
    ax_map.plot(0.0, 0.0, marker='h', color='red', markersize=10)
    ax_map.plot(locn['x'],locn['y'],marker='o',color='blue',markersize=10)
    ax_map.set_title(site)
    ax_map.set_xlabel('X (km)')
    ax_map.set_ylabel('Y (km)')


cbar = fig.colorbar(cm_ratio,cax=None,ax=fig_map.axes,location='bottom', extend='both',pad=0.05,aspect=40)
# from https://stackoverflow.com/questions/72772061/how-to-set-the-ticks-in-matplotlib-colorbar-to-be-at-the-minimum-and-maximum-val#72772248
cbar.set_ticks(levels_radar_gauge)
cbar.set_ticklabels([f'{l:.2f}' for l in levels_radar_gauge])
for f in [fig_ts,fig_map]:
    f.show()
    commonLib.saveFig(f,savedir=pathlib.Path('extra_figs'))

