# Decisions on masking
#
# 1) Mask where BBF < 0.1
# 2) Mask (per month) where count_raw_Reflectivity_thresh > 20% of  samples
# 3) Mask (per month) where number of raw samples < 70% of max samples
# 4) Also mask out where over ocn.
# Need to know how many samples there were!
# For now will get it from ds.count_Reflectivity.sel(resample_prd='1h')*24*days_in_month*6
# add name of stn, index and location to metadata of netcdf file. Make distance be in m.
# all times to be minutes since 1990-01-01

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
    result = xarray.Dataset(dict(x=x, y=y, t=time, quant=quant))
    # drop unneeded coords
    coords_to_drop = [c for c in result.coords if c not in result.dims]
    result = result.drop_vars(coords_to_drop)
    return result


def event_stats(max_precip: xarray.DataArray, max_time: xarray.DataArray, group, source: str = "CPM"):
    x_coord, y_coord = source_coords(source)
    ds = xarray.Dataset(dict(maxV=max_precip, maxT=max_time))
    grper = ds.groupby(group)
    quantiles = np.linspace(0, 1, 21)
    dataSet = grper.map(quants_locn, quantiles=quantiles, x_coord=x_coord, y_coord=y_coord).rename(quant='max_precip')
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
    if topog is None:
        expected_event_area = max_values.isel(resample_prd=1).count()
    else:
        expected_event_area = len(max_values.time) * (topog > 0).sum()
    for roll in max_values['resample_prd'].values:
        dd = event_stats(max_values.sel(resample_prd=roll),
                         max_times.sel(resample_prd=roll),
                         grp.sel(resample_prd=roll), source=source
                         ).sel(EventTime=slice(1, None))
        # at this point we have the events. Check that the total cell_count. Should be no_years*no non-nan cells in seasonalMax
        assert (int(dd.count_cells.sum('EventTime').values) == expected_event_area)
        event_time_values = np.arange(0, len(dd.EventTime))
        dd = dd.assign_coords(resample_prd=roll, EventTime=event_time_values)
        logger.debug('Computed event stats')
        # Fro temperatures time to be the same as the extreme times.
        if temp is not None:
            tc = np.array([f"{int(y)}-06-01" for y in dd.t.isel(quantv=0).dt.year])
            temp_extreme_times = temp.interp(time=tc).rename(dict(time='EventTime'))
            # convert EventTime to an index.
            temp_extreme_times = temp_extreme_times.assign_coords(resample_prd=roll, EventTime=event_time_values)
            dd['CET'] = temp_extreme_times  # add in the CET data.
            logger.debug('Added CET in')
        # add in hts
        if topog is not None:
            coords = source_coords(source)
            sel = dict(zip(coords, [dd.x, dd.y]))
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


sitecoords: tuple[float, float, float] = (144.7555, -37.8553, 45.)  # from stn metadata file.
radar = xarray.load_dataset("/home/z3542688/data/aus_rain_analysis/radar_reflectivity/Melbourne_summary_refl.nc")
bbf = xarray.load_dataarray("/home/z3542688/data/aus_rain_analysis/SRTM_Data/cbb_melbourne_grid.nc")
bbf = bbf.coarsen(x=4, y=4, boundary='trim').mean()
for c in ['x', 'y']:
    radar[c] = radar[c] * 1000.
# keep where BBF < 0.9
msk = bbf < 0.9
# and where samples > 70% of max samples.
samples = radar.count_Reflectivity.sel(resample_prd='1h') * 12
max_samples = radar.time.dt.days_in_month * 24 * 12
fraction = samples / max_samples
msk = msk & (fraction > 0.7)
# and where count_raw_Reflectivity_thresh < 20% of samples
msk = msk & (radar.count_raw_Reflectivity_thresh < 0.2 * samples)
# now get in the max reflectivity
L = radar.time.dt.season == 'JJA'
mx = radar.mean_Reflectivity.where(msk).where(L, drop=True).load()
mx = mx - np.log10(1 - bbf)  # correct for beam blockage
mxTime = radar.time_max_Reflectivity.where(msk).where(L, drop=True).load()
grp = np.floor(((mxTime - np.datetime64('1970-01-01T10:00')) / np.timedelta64(1, 'D'))).rename('EventTime')
radar_events = comp_events(mx, mxTime, grp, source='RADAR')

## plot data
proj = ccrs.TransverseMercator(*sitecoords[0:2])  # centred on the radar
fig, axs = plt.subplots(num='radar_samples', clear=True, figsize=(8, 7), subplot_kw=dict(projection=proj),
                        layout='constrained', squeeze=False, nrows=2, ncols=2
                        )

for time, ax in zip([0, 25, 50, 74], axs.flatten()):
    cm = mx.isel(time=time, resample_prd=1).plot(ax=ax, add_colorbar=False, vmin=15, vmax=60)
    ax.set_title(f'{mx.time.isel(time=time).dt.strftime("%Y-%m").values}')
    ax.coastlines()
fig.colorbar(cm, ax=axs.ravel(), orientation='horizontal', fraction=0.05, pad=0.04)
fig.show()
