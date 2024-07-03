#!/usr/bin/env python
# Reads in seasonally grouped radar data and computes the extreme events.
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
        extras: typing.Optional[typing.List[xarray.DataArray]] = None,
        source: str = 'CPM'
):
    if extras is None:
        extras = []
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
        my_logger.debug('Computed event stats')
        # deal with the extras!
        for da in extras:
            try:
                da_extreme_times = da.sel(resample_prd=roll).squeeze(drop=True).sel(time=dd.t, method='nearest')
            except KeyError:
                da_extreme_times = da.sel(time=dd.t, method='nearest')  #.rename(dict(time='EventTime'))

            dd = dd.merge(da_extreme_times.drop_vars('time'))
            my_logger.debug(f'Added {da_extreme_times.name} in')
        # add in hts
        if topog is not None:
            coords = source_coords(source)
            sel = dict(zip(coords, [dd.xpos, dd.ypos]))
            ht = topog.sel(**sel)
            my_logger.debug('Included ht')
            # drop unneeded coords
            coords_to_drop = [c for c in ht.coords if c not in ht.dims]
            ht = ht.drop_vars(coords_to_drop)
            dd['height'] = ht
        dd_lst.append(dd)
        my_logger.info(f"Processed resample_prd: {roll}")

    event_ds = xarray.concat(dd_lst, dim='resample_prd')
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
    ausLib.add_std_arguments(parser,dask=False)  # add on the std args. Turn of dask as this is rather I/O bound.
    args = parser.parse_args()
    my_logger = ausLib.process_std_arguments(args) # setup the logging and do std stuff


    extra_attrs = dict(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.utcnow().isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()]
                       )

    variable = 'rain_rate'  # set to reflectivity for reflectivity data
    in_file = args.input_file
    out_file = args.output
    out_file.parent.mkdir(exist_ok=True, parents=True)
    if out_file.exists() and (not args.overwrite):
        my_logger.warning(f"Output file {out_file} exists and overwrite not set. Exiting")
        sys.exit(0)

    radar = xarray.open_dataset(in_file).load()
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
    attrs=obs_temperature.attrs.copy()
    obs_temperature = obs_temperature.to_xarray().rename('ObsT').rename(dict(date='time')).assign_attrs(attrs)

    # want the topography
    regn = ausLib.extract_rgn(radar)
    if args.cbb_dem_files:
        cbb_dem_files = args.cbb_dem_files
    else:
        site_index = ausLib.site_numbers[site]
        cbb_dem_files = list((ausLib.data_dir / f'site_data/{site}').glob(f'{site}_{site_index:03d}_[0-9]_*cbb_dem.nc'))
        my_logger.info(f"Inferred CBB/DEM files are: {cbb_dem_files}")

    CBB_DEM = xarray.open_mfdataset(cbb_dem_files, concat_dim='prechange_start', combine='nested').sel(**regn)
    # important to select to region before coarsening for consistency with radar data processing.
    topog = CBB_DEM.elevation.max('prechange_start').coarsen(x=4, y=4, boundary='trim').mean()
    # and extract the radar data
    mx = radar[f'max_{variable}']
    mxTime = radar[f'time_max_{variable}']
    # utc hour of 14:00 corresponds to 00 in Eastern Australia.
    ref_time = '1970-01-01T14:00'
    grp = np.floor(((mxTime - np.datetime64(ref_time)) / np.timedelta64(1, 'D'))).rename('EventTime').compute()
    radar_events = comp_events(mx, mxTime, grp, source='RADAR', topog=topog,
                               extras=[obs_temperature, radar.fraction,
                                       (radar.sample_resolution.dt.seconds / 60.).rename('sample_resolution')]
                               )
    radar_events['Observed_temperature']=obs_temperature
    #TODO  add in the obs_temperature and radar.fraction timeseries
    radar_events.attrs = radar.attrs
    ausLib.write_out(radar_events,out_file,extra_attrs=extra_attrs,time_dim=None)
    

