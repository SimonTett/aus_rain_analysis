# calibrate radar data. Code needs cleaning up.
##
import pathlib
import typing


import xarray
import pathlib
import numpy as np
import cartopy.crs as ccrs
import cartopy.geodesic
import pandas as pd
import matplotlib.pyplot as plt
import ausLib

def read_gauge_data(gauge_metadata: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name, series in gauge_metadata.iterrows():
        records.append(ausLib.read_gsdr_data(series))
    result = pd.concat(records, axis=1).sort_index()
    return result
##
site='Melbourne'
files = list(pathlib.Path(f'/scratch/wq02/st7295/test/{site}_coord').glob("hist_gndrefl*rain.nc"))
files = '/scratch/wq02/st7295/summary_reflectivity_75max/Melbourne_rain_coord/hist_gndrefl_1997_*rain.nc'
radar = xarray.open_mfdataset(files) # rainfall for stations close to radar
# and convert to a dataframe
radar_resamp=radar.rain_rate.resample(time='1h',closed='right',label='right').mean().compute()
radar_df = radar_resamp.to_dataframe().unstack('station').loc[:, radar.rain_rate.name]
radar_df.index = pd.to_datetime(radar_df.index,utc=True)
meta_data = ausLib.read_gsdr_csv(pathlib.Path(f'meta_data/{site}_close.csv'))
gauge_data = read_gauge_data(meta_data).loc['1997-01-01':'1998-12-31']

all_df = pd.concat([radar_df[None:gauge_data.index.max()].stack().rename('radar_rain'),
                    gauge_data.stack().rename('gauge')], axis=1)
L = all_df.notna().all(axis=1)  # all points OK
all_df = all_df[L]
# mask out anything where both are small (< 0.1)
L = (all_df > 0.5).all(axis=1)
all_df[L].plot.scatter(x='radar_rain',y='gauge',s=4)
plt.show()
breakpoint()
## old code from here. May be useful so not deleting...
type_time_range = typing.Tuple[typing.Optional[str], typing.Optional[str]]


def read_gauge_metadata(radar_coords: np.ndarray,
                        radius: float,
                        time_range: type_time_range = (None, None)) -> pd.DataFrame:
    """
    Read the gauge metadata file, extract the gauges within the radius of the radar
    :param radar_coords: coords as numpy array (longitude, latitude) of the radar
    :param radius: radius in m
    :param time_range: time range to extract the gauges.
        Extracted if the start date is greater than the first element and the end date is less than the second element.
    :return: pd.DataFrame
    """
    #TODO -- when ausLib has a root directory use that so can run this code from anywhere.
    all_metadata = ausLib.read_gsdr_csv("meta_data/AU_GSDR_metadata.csv")
    earth_geo = cartopy.geodesic.Geodesic()
    pts = np.array([all_metadata.Longitude, all_metadata.Latitude]).T
    dist = earth_geo.inverse(radar_coords, pts)
    L = (dist[:, 0] < radius)
    with pd.option_context("mode.copy_on_write", True):
        radar_metadata = all_metadata[L]  # extract to  points close enough
        radar_metadata['Distance'] = dist[L, 0]
    if time_range[0] is not None or time_range[1] is not None:
        # filter on time.
        L = pd.Series(True, index=radar_metadata.index)
        if time_range[0] is not None:
            L = L & (radar_metadata['End_time'] >= time_range[0])  # Finish *after* the desired start date
        if time_range[1] is not None:
            L = L & (radar_metadata['Start_time'] <= time_range[1])  # Start *before* the desired end date
        radar_metadata = radar_metadata[L]
    return radar_metadata





def gauge_max(gauge_metadata: pd.DataFrame,
              variable: typing.Literal['max_rain', 'time_max_rain', 'total_rain'] = 'max_rain') -> pd.DataFrame:
    """
    Read in gauge data and compute the monthly maxima.
    :param variable: Which variable to extract. Either 'max' or 'time_max'
    :param gauge_metadata: metadata dataframe
    :return:
    """

    process_gauge = []
    my_logger.info("Extracting gauge data")
    for name, series in gauge_metadata.iterrows():
        gauge_data = ausLib.read_gsdr_data(series)
        mx = ausLib.process_gsdr_record(gauge_data, 'MS')[variable].rename(name)

        process_gauge.append(mx)
    process_gauge = pd.concat(process_gauge, axis=1)
    return process_gauge





def gauge_colocate_time(gauge_metadata: pd.DataFrame, times: pd.DataFrame) -> pd.DataFrame:
    """
    Read in gauge data and extract the times in times.
    :param gauge_metadata: dataframe of gauge metadata
    :param times: dataframe of times to extract. columns of times contain the times to extract.
    :return:extract dataframe
    """

    process_gauge = []
    my_logger.info("Extracting times from gauge data")
    for name, series in gauge_metadata.iterrows():
        gauge_data = ausLib.read_gsdr_data(series)
        gauge_data = gauge_data.reindex(times[name].dt.strftime('%Y-%m-%d %H'))
        # reset times to the same as the times dataframe
        gauge_data.index = times.index
        process_gauge.append(gauge_data)
    process_gauge = pd.concat(process_gauge, axis=1)
    return process_gauge


def extract(df: pd.DataFrame,
            start_end: typing.Tuple[str | None, str | None],
            season: typing.Optional[typing.Literal['DJF', 'MAM', 'JJA', 'SON']] = None) \
        -> pd.DataFrame:
    """
    Sub-sample data frame to time-range and season.
    :param df: dataframe to have data extracted from
    :param start_end: tuple of start and end dates. If None then no limit.
    :param season: season to extract. One of DJF, MAM, JJA, SON (or None).
    If None no season is extracted
    :return: sub-sampled dataframe
    """
    result = df[start_end[0]:start_end[1]]
    if season is not None:
        months = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}.get(season)
        L = result.index.month.isin(months)
        result = result[L]
    return result


def extract_df(data_array: xarray.DataArray,
               proj_radar: ccrs.Projection,
               gauges_close: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the radar data to the gauge locations and convert to a dataframe.
    Fix time-coords.
    :param data_array: data array to be extracted
    :param proj_radar: projection for radar data
    :param gauges_close:  metadata for gauges close!
    :return: dataframe of radar data at the gauge locations
    """
    match = radar_gauge_match(data_array, proj_radar, gauges_close, tolerance=2000.).to_dataframe()
    match = match.unstack('station').loc[:, data_array.name]
    match.index = match.index.to_period('M').to_timestamp()
    match.index = pd.to_datetime(match.index, utc=True)  # and make it UTC
    match.index.name = 'time'
    return match


##
site = 'Melbourne'
my_logger = ausLib.my_logger
ausLib.init_log(my_logger, 'DEBUG')
in_radar = list(pathlib.Path(f'/scratch/wq02/st7295/summary_reflectivity_75max/{site}').glob("hist_gndrefl*.nc"))
radar = xarray.open_mfdataset(in_radar).sel(resample_prd='1h', time=slice('1997-01-01', '2015-07-31'))
radar = ausLib.add_long_lat_coords(radar)  # add longitude & latitude to the radar data
proj_radar = ausLib.radar_projection(radar.proj.attrs)
season = 'DJF'
if season is not None:
    radar = radar.where(radar.time.dt.season == 'DJF', drop=True)
max_ref = radar.max_reflectivity
L = max_ref > 0  # drop zero reflectivity values
max_ref = max_ref.where(L)  # drop zero reflectivity values
max_time = radar.time_max_reflectivity.where(L)
# get gauges close to radar
radar_coords = np.array([radar.proj.attrs['longitude_of_central_meridian'],
                         radar.proj.attrs['latitude_of_projection_origin']])
gauges_close = read_gauge_metadata(radar_coords, 75e3,
                                   time_range=('1997-01-01', '2022-12-31'))
# select points from radar.
radar_match = extract_df(max_ref, proj_radar, gauges_close)
radar_match_time = extract_df(max_time, proj_radar, gauges_close)
# get the gauge data
gauge_mx = gauge_max(gauges_close)
gauge_mx_time = gauge_colocate_time(gauges_close, radar_match_time)
# extract to 1997 on and for DJF
gauge_mx = extract(gauge_mx, ('1997-01-01', None), season=season)
gauge_mx_time = extract(gauge_mx_time, ('1997-01-01', None), season=season)

# join the radar and gauge data together
all_df = pd.concat([radar_match[None:gauge_mx.index.max()].stack().rename('radar_reflect'),
                    gauge_mx.stack().rename('gauge')], axis=1)
L = all_df.notna().all(axis=1)  # all points OK
all_df = all_df[L]
# sort the data. Doing this as basically comparing distributions
# and the colocated data
all_df_colocate = pd.concat([
    radar_match[None:gauge_mx_time.index.max()].stack().rename('radar_reflect'),
    gauge_mx_time.stack().rename('gauge')], axis=1)
L = all_df_colocate.notna().all(axis=1)  # all points OK
all_df_colocate = all_df_colocate[L]

## now to do the stat fits
from statsmodels.api import RLM, OLS

fits_rlm = dict()
fits_ols = dict()
data_sets = dict(max=all_df, colocated=all_df_colocate)
data_fit = dict()
formula = 'np.log(gauge) ~ np.log(radar_reflect)'
min_fit = 2.

for name, data in data_sets.items():
    L = data.loc[:, 'gauge'] > min_fit  # at least min_fit mm/hr max...
    fit_data = data[L]
    if 'colocated' not in name:
        fit_data = fit_data.apply(np.sort).reset_index(drop=True)  # want to see cum dist and fit to that
    data_fit[name] = fit_data
    fits_rlm[name] = RLM.from_formula(formula, data=fit_data).fit()
    fits_ols[name] = OLS.from_formula(formula, data=fit_data).fit()

##  plot things
fig, axs = plt.subplots(nrows=2, ncols=2, num='gauge_Z_calibrate', clear=True,
                        layout='constrained', figsize=(8, 8), sharex='col', sharey='row')
for ax, (name, data) in zip(axs, data_fit.items()):
    mx = data.max()
    for a, fit, fit_type in zip(ax, [fits_rlm[name], fits_ols[name]], ['RLM', 'OLS']):
        pwr = fit.params["np.log(radar_reflect)"]
        scale = np.exp(fit.params["Intercept"])
        stat_model_str = f'R={scale:4.3f}Z' + '$^{' + f'{pwr:.3f}' + '}$'
        radar_predict = pd.Series(np.geomspace(1., mx.iloc[0], 100)).rename('radar_reflect')
        gauge_predict = np.exp(fit.predict(radar_predict))
        data.plot.scatter(y='gauge', x='radar_reflect', s=4, ax=a)
        a.plot(radar_predict, gauge_predict, color='k', linewidth=2, linestyle='dashed')
        a.set_ylabel("Gauge Monthly Rx1h (mm/h)", fontsize='small')
        a.set_xlabel("Radar Reflect Rx1h (mm$^6$/m$^3$)", fontsize='small')
        a.set_title(name + " " + fit_type + " Fit: " + stat_model_str)
        a.axhline(min_fit, color='r', linestyle='dashed')
        a.set_yscale('log')
        a.set_xscale('log')
