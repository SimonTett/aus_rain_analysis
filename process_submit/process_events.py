#!/usr/bin/env python
# Reads in seasonally grouped radar data and computes the extreme events.
# code rather cruddy!
import argparse
import multiprocessing
import pathlib
import sys

import pandas as pd

import dask
import xarray
import numpy as np
import typing

import ausLib

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
    """ compute quantiles and locations for a data set. Should contain max_value and max_time
    :param data_set: The data set to compute the quantiles and locations for.
    :param dimension: The dimension to compute the quantiles over.
    :param quantiles: The quantiles to compute.
    :param x_coord: The x co-ordinate name
    :param y_coord: The y co-ordinate name
    Returns a dataset with the quantile values of  MaxV and x, y & time values of those quantiles.
    """
    data_array = data_set.max_value  # maximum values
    time_array = data_set.max_time  # time when max occurs
    quant = data_array.quantile(quantiles, dim=dimension).rename(quantile="quantv").rename('max_value')
    order_values = data_array.values.argsort()  # indices that sort the data
    oindices = ((data_array.size - 1) * quantiles).astype('int64')  # actual indices for the rough quantile
    indices = order_values[oindices]  # actual indices in the data for the rough quantiles
    indices = xarray.DataArray(indices, coords=dict(quantv=quantiles))  # and give them co-ords so fancy indexing works
    y = data_array[y_coord].broadcast_like(data_array)[indices]  # get x-coord
    x = data_array[x_coord].broadcast_like(data_array)[indices]  # get y-coord
    time = time_array[indices]  # get time-coord
    result = xarray.Dataset(dict(xpos=x, ypos=y, t=time, max_value=quant))  # group up into to a data array
    # drop unneeded coords
    coords_to_drop = [c for c in result.coords if c not in result.dims]
    result = result.drop_vars(coords_to_drop)
    return result


def event_stats(data_set: xarray.Dataset,
                group_fn: typing.Optional[typing.Callable] = None,
                quantiles: typing.Optional[np.ndarray] = None,
                x_coord: str = 'x', y_coord: str = 'y'):
    """
    Compute the event statistics for the max_value and max_time data.
    Args:
        data_set : The dataset containing the max_value and max_time data.
        group_fn : function which generates grouping.
        quantiles : The quantiles to compute in each event.
        x_coord: the name of the x-coord
        y_coord: the name of the y-coord

    Returns:
        A dataset containing the event statistics. See quants_locn for what is returned.
        In addition, the event size and number of events are included.

    """
    if group_fn is None:
        raise ValueError("group_fn must be provided")
    if quantiles is None:
        quantiles = np.linspace(0, 1, 21)
    my_logger.debug(f'Computing events for {data_set.resample_prd.values}')
    expected_event_count = data_set['max_value'].notnull().sum().astype('int64')
    group = group_fn(data_set['max_time'])
    grper = data_set.groupby(group)
    dataSet = grper.map(quants_locn, quantiles=quantiles, x_coord=x_coord, y_coord=y_coord).dropna('EventTime')  # drop the nans.
    # group by the grouping and then apply quants_locn to each group.
    dataSet['count_cells'] = grper.count()['max_value'].astype('int64')

    # at this point we have the events. Check that the total cell_count is as expected
    assert (dataSet['count_cells'].sum() == expected_event_count)
    # set the eventTimes sensibly.
    event_time_values = np.arange(0, len(dataSet.EventTime))
    dataSet = dataSet.assign_coords(EventTime=event_time_values)
    dataSet['Event_count'] = len(dataSet.EventTime)  # store the count of the number of events

    return dataSet





def comp_events(data_set: xarray.Dataset, group_fn: typing.Optional[typing.Callable] = None,
                topog: typing.Optional[xarray.DataArray] = None,
                extras: typing.Optional[typing.List[xarray.DataArray]] = None,
                source: str = 'RADAR'
                ):
    """
    Compute the events from a dataset
    :param data_set -- must contain max_values and max_times. max_values are grouped into events. max_times used to provide times
    :param group_fn:  function applied to data_set.max_time to generate event grouping
    :param topog: Topography data
    :param extras: Any extras to be wrapped into the dataset. The **nearest** times (in months) in this dataset
        will be used to match the event times.
    :param source: Source -- used to determine the spatial co-coords to use
    :return: dataset of event data.
    """
    if extras is None:
        extras = []
    coords = source_coords(source)
    event_ds = data_set.groupby(data_set.resample_prd).map(event_stats,
                                                           group_fn=group_fn,x_coord=coords[0], y_coord=coords[1])

    # groupby resample_prd. event_stats takes a dataset and a group_fn and returns the event dataset.
    time_quantisation = 'datetime64[M]'
    time_coord = 'time'
    msk  = event_ds.max_value.notnull()
    # deal with the extras
    if extras:
        time = event_ds.t.astype(time_quantisation)
        # and then where missing set to min time,
        time = time.where(msk,np.datetime64('1970-01-01T00:00:00'))
        sel = dict(time=time, method='nearest')
        for da in extras:
            da[time_coord] = da.copy()[time_coord].astype(time_quantisation)  # quantize t
            my_logger.debug(f'Quantizing {da.name} to month')
            # extracting values and then mask and drop time coord
            da_extreme_times = da.squeeze(drop=True).sel(**sel).where(msk).drop_vars(time_coord)
            event_ds = event_ds.merge(da_extreme_times)
            my_logger.debug(f'Added {da_extreme_times.name} in')
    # add in hts
    if topog is not None:
        xpos = event_ds.xpos.where(msk,event_ds.xpos.min())
        ypos = event_ds.ypos.where(msk,event_ds.ypos.min())
        coords = source_coords(source)
        sel = dict(zip(coords, [xpos, ypos]))
        ht = topog.sel(**sel).where(msk)
        my_logger.debug('Included ht')
        # drop unneeded coords
        coords_to_drop = [c for c in ht.coords if c not in ht.dims]
        ht = ht.drop_vars(coords_to_drop)
        event_ds['height'] = ht



    return event_ds


acorn_lookup = dict(Adelaide=23000, Melbourne=86338, Wtakone=96003, Sydney=66214, Brisbane=40842, Canberra=70351,
                    Cairns=31011, Mornington=29077,
                    Grafton=59151, Newcastle=61078, Gladstone=39083)
# temp stations for sites. See www.bom.gov.au/climate/data/acorn-sat
if __name__ == '__main__':
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!
    parser = argparse.ArgumentParser(description="Compute events for Australian radar data")
    parser.add_argument('input_file', type=pathlib.Path, help='input file for seasonal processed radar data')
    parser.add_argument('output', type=pathlib.Path, help='Filename for events file. ')
    parser.add_argument('--station_id', type=int, help='ACORN id for station used to generate temperature covariate. '
                                                       'If not provided computed from site in input file'
                        )
    parser.add_argument('--cbb_dem_files', nargs='+', type=pathlib.Path,
                        help='filenames for the beam blockage dem ancillary files. '
                             'If not provided computed from site  in input file'
                        )
    parser.add_argument('--site', type=str, help='site name for radar data. If not provided taken from input file')
    parser.add_argument('--region', type=float, nargs=4, help='Region: x0, y0, x1, y1. Default is whole dataset')
    ausLib.add_std_arguments(parser,dask=False)  # add on the std args.
    # Turn of dask as this is rather I/O. Turning it on slows things down!
    args = parser.parse_args()
    my_logger = ausLib.process_std_arguments(args)  # setup the logging and do std stuff

    extra_attrs = dict(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.utcnow().isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()]
                       )
    if args.region:
        extra_attrs.update(region=args.region)

    variable = 'rain_rate'  # set to reflectivity for reflectivity data
    in_file = args.input_file
    out_file = args.output
    out_file.parent.mkdir(exist_ok=True, parents=True)
    if out_file.exists() and (not args.overwrite):
        my_logger.warning(f"Output file {out_file} exists and overwrite not set. Exiting")
        sys.exit(0)

    radar = xarray.open_dataset(in_file,chunks=dict(resample_prd=1,time=-1,x=-1,y=-1))
    regn = None
    if args.region:
        rgn = args.region
        regn = dict(x=slice(rgn[0], rgn[2]), y=slice(rgn[1], rgn[3]))
        radar = radar.sel(**regn)

    site = radar.attrs.get('site', None)
    if args.site:
        site = args.site
        extra_attrs.update(site=site)  # update the site
        my_logger.warning(f"Forcing site to be {site}")
    else:
        site = radar.attrs.get('site', None)
        my_logger.info(f"Site from metadata: {site}")
        if site is None:
            raise ValueError("No site name given or in metadata. Specify via --site option")

    # Get data for temperature covariate.
    station_id = args.station_id
    if station_id is None:
        station_id = acorn_lookup[site]
        my_logger.debug('No station id provided. Using default')
    my_logger.info(f'Using acorn id of {station_id}')
    extra_attrs.update(station_id=station_id)
    # get the temperature data for the site.
    obs_temperature = ausLib.read_acorn(station_id, what='mean').resample('QS-DEC').mean()
    # get times to middle of season.
    offset = (obs_temperature.index.diff() / 2.).fillna(pd.Timedelta(45, 'D'))
    obs_temperature.index = obs_temperature.index + offset
    attrs = obs_temperature.attrs.copy()
    obs_temperature = obs_temperature.to_xarray().rename('ObsT').rename(dict(date='time')).assign_attrs(attrs)

    # want the topography
    orig_regn = ausLib.extract_rgn(radar)
    if args.cbb_dem_files:
        cbb_dem_files = args.cbb_dem_files
    else:
        site_index = ausLib.site_numbers[site]
        cbb_dem_files = list((ausLib.data_dir / f'site_data/{site}').glob(f'{site}_{site_index:03d}_[0-9]_*cbb_dem.nc'))
        my_logger.info(f"Inferred CBB/DEM files are: {cbb_dem_files}")

    CBB_DEM = xarray.open_mfdataset(cbb_dem_files, concat_dim='prechange_start', combine='nested').sel(**orig_regn)
    # important to select to region before coarsening for consistency with radar data processing.
    topog = CBB_DEM.elevation.max('prechange_start').coarsen(x=4, y=4, boundary='trim').mean()
    if regn is not None:  # select to the region requested here.
        topog = topog.sel(**regn).squeeze(drop=True)

    # and extract the radar data
    mx = radar[f'max_{variable}']
    mxTime = radar[f'time_max_{variable}']
    # Time is end of period -- adjust to middle of period by subtracting 1/2 the resample_prd
    offset = xarray.apply_ufunc(lambda ta: [pd.Timedelta(str(v)) / 2 for v in ta], mxTime.resample_prd)
    mxTime = mxTime - offset
    # utc hour of 14:00 corresponds to roughly 00 in Eastern Australia.
    ref_time = '1970-01-01T14:00'

    ds = xarray.Dataset(dict(max_value=mx, max_time=mxTime))
    group_fn = lambda time  : np.floor(((time - np.datetime64(ref_time)) / np.timedelta64(1, 'D'))).rename('EventTime')#.compute()
    radar_events = comp_events(ds, source='RADAR', topog=topog, group_fn=group_fn,
                               extras=[obs_temperature, radar.fraction,
                                       (radar.sample_resolution.dt.seconds / 60.).rename('sample_resolution')]
                               )
    radar_events['Observed_temperature'] = obs_temperature
    radar_events.attrs = radar.attrs
    radar_events = radar_events.compute()
    ausLib.write_out(radar_events, out_file, extra_attrs=extra_attrs, time_dim=None)
