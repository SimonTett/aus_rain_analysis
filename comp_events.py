# Reads in seasonally grouped radar data and computes the extreme events.

import xarray
import wradlib as wrl
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import typing

import ausLib

logger = ausLib.my_logger
ausLib.init_log(logger, level='INFO')
horizontal_coords = ['x', 'y']  # for radar data.
cpm_horizontal_coords = ['grid_latitude', 'grid_longitude']  # horizontal coords for CPM data.


def source_coords(source: str) -> tuple[str, ...]:
    """
    Return coords for different sources.
    :param source:  Source -- CPM or RADAR
    :return: co-ord names s a tuple.
    :rtype:
    """
    if source == 'CPM':
        return tuple(cpm_horizontal_coords)
    elif source == 'RADAR':
        return tuple(horizontal_coords)
    else:
        raise ValueError(f"Unknown source {source}")


def quants_locn(
        data_set: xarray.Dataset,
        dimension: typing.Optional[typing.Union[str, typing.List[str]]] = None,
        quantiles: typing.Optional[typing.Union[np.ndarray, typing.List[float]]] = None,
        x_coord: str = 'grid_longitude',
        y_coord: str = 'grid_latitude'
) -> xarray.Dataset:
    """ compute quantiles and locations"""
    if quantiles is None:
        quantiles = np.linspace(0.0, 1.0, 6)

    data_array = data_set.maxV  # maximum values
    time_array = data_set.maxT  # time when max occurs
    quant = data_array.quantile(quantiles, dim=dimension).rename(quantile="quantv")
    order_values = data_array.values.argsort()
    oindices = ((data_array.size - 1) * quantiles).astype('int64')
    indices = order_values[oindices]  # actual indices to the data where the quantiles are roughly
    indices = xarray.DataArray(indices, coords=dict(quantv=quantiles))  # and give them co-ords
    y = data_array[y_coord].broadcast_like(data_array)[indices]
    x = data_array[x_coord].broadcast_like(data_array)[indices]
    time = time_array[indices]
    result = xarray.Dataset(dict(xpos=x, ypos=y, t=time, quant=quant))
    # drop unneeded coords
    coords_to_drop = [c for c in result.coords if c not in result.dims]
    result = result.drop_vars(coords_to_drop)
    return result


def event_stats(max_value: xarray.DataArray, max_time: xarray.DataArray, group, source: str = "CPM"):
    x_coord, y_coord = source_coords(source)
    ds = xarray.Dataset(dict(maxV=max_value, maxT=max_time))
    grper = ds.groupby(group)
    quantiles = np.linspace(0, 1, 21)
    dataSet = grper.map(quants_locn, quantiles=quantiles, x_coord=x_coord, y_coord=y_coord).rename(quant='max_value')
    count = grper.count().maxV.rename("# Cells")
    dataSet['count_cells'] = count
    return dataSet


def comp_events(
        max_values: xarray.DataArray,
        max_times: xarray.DataArray,
        grp: xarray.DataArray,
        topog: typing.Optional[xarray.DataArray] = None,
        temp: typing.Optional[xarray.DataArray] = None,
        source: str = 'CPM'
):
    dd_lst = []
    for roll in max_values['resample_prd'].values:
        mx = max_values.sel(resample_prd=roll)
        expected_event_count = mx.notnull().sum().astype('int64')
        dd = event_stats(mx,
                         max_times.sel(resample_prd=roll),
                         grp.sel(resample_prd=roll).fillna(0.0), source=source
                         ).sel(EventTime=slice(1, None))
        # at this point we have the events. Check that the total cell_count is as expected
        assert (int(dd.count_cells.sum('EventTime').values) == expected_event_count)
        event_time_values = np.arange(0, len(dd.EventTime))
        dd = dd.assign_coords(resample_prd=roll, EventTime=event_time_values)
        logger.debug('Computed event stats')
        # For temperatures time to be the same as the extreme times.
        if temp is not None:
            temp_extreme_times = temp.sel(time=dd.t,method='nearest')#.rename(dict(time='EventTime'))
            dd['temp'] = temp_extreme_times.drop_vars('time')  # add in the temperature data.
            logger.debug('Added temp in')
        # add in hts
        if topog is not None:
            coords = source_coords(source)
            sel = dict(zip(coords, [dd.xpos, dd.ypos]))
            ht = topog.sel(**sel)
            logger.debug('Included ht')
            # drop unneeded coords
            coords_to_drop = [c for c in ht.coords if c not in ht.dims]
            ht = ht.drop_vars(coords_to_drop)
            dd['height'] = ht
        dd_lst.append(dd)
        logger.info(f"Processed resample_prd: {roll}")

    event_ds = xarray.concat(dd_lst, dim='resample_prd')
    return event_ds

site = 'Melbourne'
site_no = 86338 # station number
sitecoords: tuple[float, float, float] = (144.7555, -37.8553, 45.)  # from stn metadata file.
regn = dict(x=slice(-75e3,75e3),y=slice(-75e3,75e3)) # within 75 km of radar site
in_file= ausLib.data_dir/f"summary_reflectivity/processed/{site}_hist_gndrefl_DJF.nc"
out_file = ausLib.data_dir/f"events/{site}_hist_gndrefl_DJF.nc"
out_file.parent.mkdir(exist_ok=True,parents=True)
cbb_dem = xarray.load_dataset(ausLib.data_dir/f'ancil/{site}_cbb_dem.nc')
topog = cbb_dem.elevation.coarsen(x=4, y=4, boundary='trim').mean().sel(**regn)
# get the temperature data for the site.
obs_temperature = ausLib.read_acorn(site_no,what='mean').resample('QS-DEC').mean().to_xarray().rename(dict(date='time'))
obs_temperature=obs_temperature.where(obs_temperature.time.dt.season=='DJF',drop=True)
# and get the radar data
radar = xarray.open_dataset(in_file).load()
mx = radar.max_reflectivity.sel(**regn)
mxTime = radar.time_max_reflectivity.sel(**regn)
# utc hour of 14:00 corresponds to 00 in Australia.
ref_time = '1970-01-01T14:00'
grp = np.floor(((mxTime - np.datetime64(ref_time)) / np.timedelta64(1, 'D'))).rename('EventTime').compute()
radar_events = comp_events(mx, mxTime, grp, source='RADAR',topog=topog,temp=obs_temperature)
# finally we can save the events
radar_events.to_netcdf(out_file)
