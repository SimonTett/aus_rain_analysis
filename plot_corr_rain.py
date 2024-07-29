# produce scatter plot between implied correction from radar and gauge rain vs actual correction.
import pathlib

import ausLib
import xarray
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import commonLib
import cartopy
import cartopy.geodesic
import cartopy.crs as ccrs

def read_radar_ds(site:str, calib:str) -> xarray.Dataset:
    direct = ausLib.data_dir / f'processed/{site}_rain_{calib}'
    file = direct / f'monthly_mean_{site}_rain_{calib}.nc'
    if not file.exists():
        raise FileNotFoundError(f'No file {file}')

    ds = xarray.open_dataset(file)
    return ds
def comp_total_rain(ds:xarray.Dataset, rgn: dict[str, slice]) -> xarray.DataArray:

    mean_rain = ds.mean_raw_rain_rate.sel(**rgn)
    total_rain = (mean_rain.mean(['x', 'y']) * mean_rain.time.dt.days_in_month * 24).load()
    return total_rain


calib = 'melbourne'
sens = 'brisbane'

my_logger = ausLib.setup_log(1)
sydney_sens_studies = ['Sydney_rain_brisbane', 'Sydney_rain_melbourne',
                       'Sydney_rain_melbourne_10min', 'Sydney_rain_melbourne_5_65']
#read_plot_mean_monthly_rain = False # uncomment to force re-read of data

if not 'implied_correction' in locals():
    r = 75e3
    rgn = dict(x=slice(-r, r), y=slice(-r, r))
    obs_files = sorted(ausLib.agcd_rain_dir.glob('*.nc'))
    ds_obs = xarray.open_mfdataset(obs_files).sel(time=slice('1997', None))
    implied_correction = dict()
    correction = dict()
    implied_correction_sens = dict()
    site_info = dict()

    for site, no in ausLib.site_numbers.items():
        site_info[site] = ausLib.site_info(no)
        site_rec = site_info[site].iloc[0]

        #site_info[site] = ausLib.site_info(ausLib.site_numbers[site])
        # work  out region for gauge -- convert from m to long/l
        ds = read_radar_ds(site, calib)
        total_rain = comp_total_rain(ds, rgn)
        ds_sens  = read_radar_ds(site, sens)
        total_rain_sens = comp_total_rain(ds_sens, rgn)
        radar_proj = ausLib.radar_projection(ds.proj.attrs)
        trans_coords = ccrs.PlateCarree().transform_points(radar_proj,
                                                           x=np.array([-r, r]), y=np.array([-r,r]))
        # get in the gauge data then interpolate to radar grid and mask by radar data
        lon_slice = slice(trans_coords[0, 0], trans_coords[1, 0])
        lat_slice = slice(trans_coords[0, 1], trans_coords[1, 1])
        gauge = ds_obs.precip.sel(lon=lon_slice, lat=lat_slice)
        longitude = ds.longitude.isel(time=0).squeeze(drop=True)
        latitude =ds.latitude.isel(time=0).squeeze(drop=True)
        time = ds.time
        gauge = gauge.interp(lon=longitude, lat=latitude, time=time).where(total_rain.notnull())
        gauge_total = gauge.mean(['x', 'y']).load()
        implied_correction[site] = -10 * np.log10(gauge_total / total_rain) / ds.attrs['to_rain'][1]
        implied_correction_sens[site] = -10 * np.log10(gauge_total / total_rain_sens) / ds_sens.attrs['to_rain'][1]
        # get inthe actual correction
        meta_data_files = sorted((ausLib.data_dir / f'site_data/{site}_metadata/').glob(f'{site}_*_metadata.nc'))
        # need to check for existence to stop file system errors..
        ok = [f.exists() for f in meta_data_files]
        if not all(ok):
            FileNotFoundError(f'Not all files exist for {site}')

        ds_meta = xarray.open_mfdataset(meta_data_files, concat_dim='time', combine='nested').sortby('time')
        correction[site] = ds_meta.calibration_offset.load()
        my_logger.info(f'Loaded data for {site}')

    my_logger.info('Loaded all data')
else:
    my_logger.info('Data already loaded')

fig, axes = ausLib.std_fig_axs(f'corr_rain', sharey=True, sharex=True)
for site, ax in axes.items():
    corr = correction[site].interp(time=implied_correction[site].time)
    ax.scatter(implied_correction[site], corr, s=2,color='blue')
    # add on a regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(implied_correction[site], corr)
    #ax.plot(x, y, color='red')
    ax.axline((0, intercept), slope=slope, color='blue')
    ax.text(0.97, 0.05, f'y={intercept:3.2f}+{slope:3.2}x', transform=ax.transAxes, ha='right', va='bottom',color='blue',size='small')
    ax.scatter(implied_correction_sens[site], corr, s=2,color='red')
    slope, intercept, r_value, p_value, std_err = stats.linregress(implied_correction_sens[site], corr)
    ax.axline((0, intercept), slope=slope, color='red')
    ax.text(0.97, 0.15, f'y={intercept:3.2f}+{slope:3.2}x', transform=ax.transAxes, ha='right', va='bottom',
            color='red', size='small')
    ax.set_xlabel('Implied Corr (dBZ)')
    ax.set_xlim(-20, None)
    ax.set_ylabel('Actual Corr (dBZ)')
    ax.set_title(site)
    ax.label_outer()

commonLib.saveFig(fig,savedir=pathlib.Path('radar_corr_plots'))