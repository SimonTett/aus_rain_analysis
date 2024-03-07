# compare radar and in situ monthly 1 hr extremes.
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import pandas as pd
import pytz

import ausLib
import xarray
import pathlib
import cartopy.crs as ccrs
import cartopy.geodesic
import numpy as np
import logging

my_logger = logging.getLogger(__name__)
all_metadata = ausLib.read_gsdr_csv("AU_GSDR_metadata.csv")
yr_durn = (all_metadata.End_time - all_metadata.Start_time).dt.total_seconds() / (365.24 * 24 * 60 * 60)
all_metadata = all_metadata[yr_durn > 10]

radar_dir = pathlib.Path("/scratch/wq02/st7295/radar/")
radars = sorted(['cairns', 'gladstone', 'canberra', 'newcastle', 'wtakone', 'sydney',
                 'melbourne', 'adelaide', 'mornington', 'brisbane', 'grafton'])
#radars = ['gladstone']
for radar_name in radars:
    print("Processing ", radar_name)
    pth = radar_dir / radar_name / f"processed_{radar_name}.nc"
    radar = xarray.open_dataset(pth).isel(x=slice(50, 251), y=slice(50, 251))  # 100 km.
    # some radars have (slightly) different locations as the site moved a little bit.
    # melbourne seems to have time varying longitude
    for c in ['latitude', 'longitude']:
        if "time" in radar[c].coords:
            coord = radar[c]
            coord_sd_mx = coord.std('time').max()
            if coord_sd_mx < 1e-3:
                coord = coord.isel(time=0).drop_vars('time')
                radar = radar.assign_coords({c: coord})
                my_logger.warning(f"Fixed radar {c} coords")
            else:
                raise Exception(f"too much var in {c} coords")

    centre = [radar.attrs['origin_longitude'], radar.attrs['origin_latitude']]
    earth_geo = cartopy.geodesic.Geodesic()
    pts = np.array([all_metadata.Longitude, all_metadata.Latitude]).T
    dist = earth_geo.inverse(centre, pts)
    L = (dist[:, 0] < 100e3) & (all_metadata.End_time > '2010')  # closer than 100 km and data ending > 2010
    radar_metadata = all_metadata[L]  # extract to 100 km radius.
    gauge = dict()
    radar_match = dict()
    process_gauge = dict()
    my_logger.info("Extracting gauge data")
    lon_pts = []
    lat_pts = []
    name_pts = []
    for name, series in radar_metadata.iterrows():
        gauge_data = ausLib.read_gsdr_data(series)
        # extract to JJA and from 2001 to 2015
        gauge_data = gauge_data['2000-12-01':'2015-02-28']
        L = gauge_data.index.month.isin([12, 1, 2])
        gauge_data = gauge_data[L]
        if (
        ~gauge_data.isnull()).sum() < 5 * 90 * 24:  # at least 5 summers worth of data (5 years * 90 days * 24 hours)
            continue
        process_gauge[name] = ausLib.process_gsdr_record(gauge_data, 'MS')
        gauge[name] = gauge_data
        lon_pts.append(series.Longitude)
        lat_pts.append(series.Latitude)
        name_pts.append(name)

    if len(gauge) == 0:  # no comparison possible
        my_logger.warning(f"No gauges close enough for {radar_name}. Skipping further processing")
        continue
    # now extract the radar data -- will put this straight into a dataframe
    my_logger.info("Extracting pts from radar")
    pts = np.row_stack((lon_pts, lat_pts))
    xind, yind = ausLib.index_ll_pts(radar.longitude.values, radar.latitude.values, pts,
                                     tolerance=2e-2)  # want to roughly 1 km.
    xind = xarray.DataArray(xind, dims='station', coords=dict(station=list(gauge.keys())))
    yind = xarray.DataArray(yind, dims='station', coords=dict(station=list(gauge.keys())))
    L = radar.time.dt.season == 'DJF'  # want summer
    radar_match = radar.max_rain.where(L, drop=True).sel(time=slice('2000-12-01', '2015-02-28')).isel(x=xind,
                                                                                                      y=yind).to_dataframe()
    # and then pivot the table to make it easier to deal with. Python multi-indices are hard...
    #radar_match = pd.pivot_table(radar_match.reset_index(), aggfunc='first', index='time', columns='station',
    #                             values='max_rain')
    radar_match = radar_match.unstack('station').loc[:, 'max_rain']
    # and then make the time UTC.
    radar_match.index = pd.to_datetime(radar_match.index, utc=True)
    print(f"{radar_name} has {len(radar_match.columns)} gauges to compare with")

    mn_max_radar = radar.max_rain.where(radar.time.dt.season == 'DJF', drop=True).sel(
        time=slice('2000-12-01', '2015-02-28')).mean('time')
    rng = np.sqrt(mn_max_radar.x.astype('float') ** 2 + mn_max_radar.y.astype('float') ** 2) / 1e3
    rng = rng.assign_coords(latitude=mn_max_radar.latitude, longitude=mn_max_radar.longitude)
    ## extract the station info.
    lng = pd.Series({k: radar_metadata.loc[k].Longitude for k in process_gauge.keys()}).rename('longitude')
    latitude = pd.Series({k: radar_metadata.loc[k].Latitude for k in process_gauge.keys()}).rename('latitude')


    def comp_mean_mx(stn_series: pd.DataFrame) -> float:
        L = stn_series.index.month.isin([12, 1, 2])
        mn = stn_series[L]['2009-12-01':'2015-02-28'].max_rain.mean()
        return mn


    stn_mn = pd.Series({k: comp_mean_mx(df) for k, df in process_gauge.items()}).rename('mean_mx')
    radar_mean = radar_match.mean().rename('Radar')
    station_data = pd.concat([stn_mn, lng, latitude], axis=1)
    merge_radar_stn = pd.concat([radar_mean, stn_mn.rename('Gauge')], axis=1)
    ## plot the radar and the location of the gauges. Will do for summer months for 2010-2015
    fig, axs = plt.subplot_mosaic(mosaic=[["locations", "scatter_mn", "scatter"]],
                                  per_subplot_kw=dict(locations=dict(projection=ccrs.PlateCarree())),
                                  layout='constrained', figsize=(8, 4.5), clear=True,
                                  num=f'radar_gauge_match_{radar_name}',
                                  width_ratios=[2, 1, 1])
    import matplotlib.colors as mcolors

    scatter_ax = axs['scatter']
    loc_ax = axs['locations']
    scatter_mn_ax = axs['scatter_mn']
    # mn=np.min([stn_mn.min(),mn_max_radar.min()])
    mn = stn_mn.min()
    mn = np.floor(mn)
    # mx=np.min([,mn_max_radar.max()])
    mx = stn_mn.max()
    mx = np.ceil(mx)
    levels = np.unique(np.floor(np.linspace(mn, mx, 10)))

    kw_colorbar = dict(orientation='horizontal', fraction=0.1, aspect=40, pad=0.05, spacing='uniform', label='mm/h')
    norm = mcolors.BoundaryNorm(levels, ncolors=256, extend='both')
    cmap = 'RdYlBu'

    mn_max_radar.plot(ax=loc_ax, cmap=cmap, levels=levels, y='latitude', x='longitude', add_colorbar=False)
    # cbar_kwargs=kw_colorbar)
    rng.plot.contour(ax=loc_ax, colors='black', linestyles='solid', levels=np.arange(0, 120, 20), y='latitude',
                     x='longitude')
    # now add on the points
    station_data.plot.scatter(ax=loc_ax, x='longitude', y='latitude', marker='o', s=50,
                              c=station_data.mean_mx, norm=norm, edgecolor='k',
                              colorbar=False, cmap=cmap,
                              transform=ccrs.PlateCarree())
    loc_ax.coastlines(color='green', linewidth=2)
    g = loc_ax.gridlines(draw_labels=True)
    g.top_labels = False
    g.left_labels = False
    loc_ax.set_title("Mean of summer monthly Rx1h")
    # scatter plot the means.
    merge_radar_stn.plot.scatter(ax=scatter_mn_ax, x='Gauge', y='Radar', marker='o', s=30,
                                 c='Gauge', norm=norm, edgecolor='k', colorbar=False, cmap=cmap)
    scatter_mn_ax.set_xlabel("Gauge Mean Monthly Rx1h (mm/h)", fontsize='small')
    scatter_mn_ax.set_ylabel("Radar Mean Monthly Rx1h (mm/h)", fontsize='small')
    scatter_mn_ax.set_title("Mean Monthly Rx1h ")
    scatter_mn_ax.axline((mn, mn), (mx, mx))  # plot 1:1 line
    mx_mx = np.ceil(merge_radar_stn.loc[:, ['Gauge', 'Radar']].max().max() * 1.1)
    mn_mn = np.floor(merge_radar_stn.loc[:, ['Gauge', 'Radar']].min().min() * 0.9)
    scatter_mn_ax.set_xlim(mn_mn, mx_mx)
    scatter_mn_ax.set_ylim(mn_mn, mx_mx)
    # now for the scatter of the maxes..
    colorbar = False
    all_df = []
    for key, df in process_gauge.items():
        merge_df = pd.concat([df.max_rain.rename(f'gauge'), radar_match.loc[:, key].rename('radar')], axis=1)
        merge_df['mean_mx'] = station_data.mean_mx[key]
        L = merge_df.index.month.isin([12, 1, 2])
        merge_df = merge_df[L].dropna(axis=0)
        merge_df.plot.scatter(ax=scatter_ax, x='gauge', y='radar', marker='o', s=5,
                              c='mean_mx', norm=norm, edgecolor='k', colorbar=False, cmap=cmap)

        merge_df['source'] = key
        merge_df = merge_df.set_index('source', append=True).unstack('source')
        all_df.append(merge_df.droplevel(level='source', axis=1).reset_index(drop=True))
    all_df = pd.concat(all_df, axis=0).reset_index(drop=True)
    # add on regression -- will predict radar from gauge.
    from statsmodels.api import RLM
    mx_mx = np.ceil(all_df.max().max() * 1.1)
    formula = 'np.log(gauge) ~ np.log(radar)'
    min_fit=1.
    L = np.all(all_df.loc[:, ['gauge', 'radar']] > min_fit, axis=1)  # at least 1mm/hr max...
    rfit = RLM.from_formula(formula, data=all_df[L]).fit()
    radar_predict= pd.Series(np.geomspace(min_fit,mx_mx, 100)).rename('radar')
    gauge_predict= np.exp(rfit.predict(radar_predict))
    scatter_ax.plot(gauge_predict,radar_predict,color='k',linewidth=2,linestyle='dashed')

    scatter_ax.set_xlabel("Gauge Monthly Rx1h (mm/h)", fontsize='small')
    scatter_ax.set_ylabel("Radar Monthly Rx1h (mm/h)", fontsize='small')
    scatter_ax.set_title("Monthly Rx1h")
    scatter_ax.axline((mn, mn), (mx, mx))  # plot 1:1 line


    scatter_ax.set_xlim(1, mx_mx)
    scatter_ax.set_ylim(1, mx_mx)
    scatter_ax.set_yscale('log')
    scatter_ax.set_xscale('log')

    cm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(cm, ax=list(axs.values()), boundaries=levels, **kw_colorbar)
    fig.suptitle(f"DJF {radar_name.capitalize()} Radar/Gauges")
    fig.show()
    fig.savefig(f'figures_compare/radar_gauge_{radar_name}.png', dpi=300)
