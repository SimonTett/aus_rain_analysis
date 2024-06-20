#!/usr/bin/env python
# Mask & compute seasonal average then write out data.
#
# 1) Mask where BBF > 0.5
# 2) Mask (per month) where count_raw_Reflectivity_thresh > 20% of  samples
# 3) Mask (per month) where number of raw samples < 70% of max samples
# 4) Also mask out where over ocn.

import argparse
import sys

import dask
import xarray
import wradlib as wrl
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import typing
import pandas as pd
import pathlib
import ast
import multiprocessing

import ausLib

horizontal_coords = ['x', 'y']  # for radar data.
cpm_horizontal_coords = ['grid_latitude', 'grid_longitude']  # horizontal coords for CPM data.


def group_data_set(ds: xarray.Dataset, *args, group_dim: str = 'time', **kwargs) -> xarray.Dataset:
    """

    Args:
        ds: dataset
        group_dim: dimension to group over

    Returns: grouped dataset for variables that start with:
        max_  -- compute the max.
        mean_ -- compute the mean.
        count_ -- sum the variable.
        time_max_ -- take argmax of the equiv max_ var and then use that index.
        sample_resolution -- take the average.
        fract_ compute the mean
        time_bounds -- take the min and max values.
    Anything else -- just take the first one.

    """
    time_str = ds.time[[0, -1]].dt.strftime('%Y-%m-%d').values
    my_logger.info(
        f"Grouping over {group_dim} with args: {args} & kwargs: {kwargs} for time: {time_str} {ausLib.memory_use()}")
    result = dict()
    max_var = 'max_rain_rate'  # set to 'reflectivity' for raw radar data
    if ds[max_var].notnull().sum() == 0:
        my_logger.warning(f"No valid values for max_reflectivity at {time_str}")
        return xarray.Dataset()

    with xarray.set_options(keep_attrs=True):
        for vname, da in ds.items():
            if vname.startswith('max_'):
                da_group = da.max(group_dim)
                my_logger.debug(f"Computed max for {vname}")
            elif vname.startswith('mean_'):
                da_group = da.mean(group_dim)
                my_logger.debug(f"Computed mean for {vname}")
            elif vname.startswith('count_'):
                da_group = da.sum(group_dim)
                my_logger.debug(f"Computed sum for {vname}")
            elif vname.startswith('time_max_'):
                max_name = vname.replace('time_max_', 'max_')
                indx = ds[max_name].idxmax(group_dim).compute()
                L = indx.notnull()
                if L.sum() == 0:
                    my_logger.warning(f"No valid values for {vname}")
                    continue
                indx = indx.where(L, da[group_dim].min())
                da_group = da.sel({group_dim: indx}).compute().where(L, np.datetime64('NaT'))
                my_logger.debug(f"Computed time_max for {vname}")
            elif vname.startswith('median_'):
                da_group = da.median(group_dim)
                my_logger.debug(f"Computed median for {vname}")
            elif vname.startswith('fract_'):
                da_group = da.mean(group_dim)
                my_logger.debug(f"Computed mean for {vname}")
            elif ('_fract' in vname) or ('_fract_' in vname):
                my_logger.warning('Using old fract_ naming. Reprocess')
                da_group = da.mean(group_dim)
                my_logger.debug(f"Computed mean for {vname}")
            elif vname in ['sample_resolution', 'max_fraction', 'threshold']:
                da_group = da.mean(group_dim)
                my_logger.debug(f"Computed mean for {vname}")
            elif vname == 'time_bounds':
                da = da.load()
                da_group = xarray.concat([da.min(), da.max()], dim='bounds')
                my_logger.debug(f"Computed time_bounds for {vname}")
            else:
                my_logger.debug(f"Taking first one for {vname}")
                da_group = da.isel({group_dim: 0})
            result[vname] = da_group.drop_vars(group_dim, errors='ignore').load() # load it. Loading really speeds things up!
    result = xarray.Dataset(result).assign_attrs(ds.attrs)
    result['input_count'] = ds.time.count()  # how many months in the dataset.
    my_logger.info(f"Grouped data for {time_str} {ausLib.memory_use()}")
    return result


# Main code
if __name__ == '__main__':
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!
    parser = argparse.ArgumentParser(description="Compute Masked seasonal mean data for processed radar data")
    parser.add_argument('input_files', nargs='+', type=pathlib.Path, help='Input files for radar data')
    parser.add_argument('--output', type=pathlib.Path, help='name of output file', required=True)
    parser.add_argument('--site', type=str, help='Site name -- overrides meta data in input files')
    parser.add_argument('--cbb_dem_files', type=pathlib.Path, help='Names of CCB/DEM files', nargs='+')
    parser.add_argument('--season', type=str, help='Season to use', default='DJF')
    parser.add_argument('--years', nargs='+', help='years to use', type=int)
    ausLib.add_std_arguments(parser)  # add on the std args
    args = parser.parse_args()
    my_logger = ausLib.setup_log(args.verbose, log_file=args.log_file)  # setup the logging
    for name, value in vars(args).items():
        my_logger.info(f"Arg:{name} =  {value}")
    if args.dask:
        my_logger.info('Starting dask client')
        client = ausLib.dask_client()
        my_logger.warning('Use of dask likely slows this down.')
    else:
        dask.config.set(scheduler="single-threaded")  # make sure dask is single threaded.
        my_logger.info('Running single threaded')

    out_radar = args.output
    if out_radar.exists() and (not args.overwrite):
        my_logger.warning(f"Output file {out_radar} exists and overwrite not set. Exiting")
        sys.exit(0)
    out_radar.parent.mkdir(exist_ok=True, parents=True)  # make directory if it doesn't exist.

    in_radar = sorted(args.input_files)
    for file in in_radar:
        if not file.exists():
            raise FileNotFoundError(f"Input file {file} not found")
    radar = xarray.open_mfdataset(in_radar, parallel=True)
    extra_attrs = dict(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.utcnow().isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()])
    # add in the original region to the extra_attrs/program_args
    rgn = [v for v in radar.attrs['program_args'] if v.startswith('region:')][0]
    extra_attrs['program_args'].append(rgn)
    if args.site:
        site = args.site
        extra_attrs.update(site=site)  # update the site
    else:
        site = radar.attrs.get('site', None)
        my_logger.info(f"Site from metadata: {site}")
        if site is None:
            raise ValueError("No site name given or in metadata. Specify via --site option")

    if args.cbb_dem_files:
        cbb_dem_files = args.cbb_dem_files
    else:
        site_index = ausLib.site_numbers[site]
        cbb_dem_files = list((ausLib.data_dir / 'SRTM_Data/radar_strm').glob(f'{site_index:03d}_[0-9]_*cbb_dem.nc'))
        my_logger.info(f"Inferred CBB/DEM files are: {cbb_dem_files}")

    variable_name = 'rain_rate'

    # select season we want
    L = radar.time.dt.season == args.season
    if args.years:
        L = L & (radar.time.dt.year.isin(args.years))

    radar = radar.where(L, drop=True)
    my_logger.info(radar.proj.dims)
    # extract the region used from the meta-data
    regn = ausLib.extract_rgn(radar)

    CBB_DEM = xarray.open_mfdataset(cbb_dem_files, concat_dim='prechange_start', combine='nested').sel(**regn)
    CBB_DEM = CBB_DEM.max('prechange_start').coarsen(x=4, y=4, boundary='trim').mean()
    bbf = CBB_DEM.CBB.clip(0, 1)
    # keep where BBF < 0.5 and land.
    msk = (bbf < 0.5) & (CBB_DEM.elevation > 0.0)
    # msk variables which have both an x and y dimension.
    xy_vars = [v for v in radar.variables if ('x' in radar[v].dims) and ('y' in radar[v].dims)]
    for var in xy_vars:
        radar[var] = radar[var].where(msk)
        my_logger.debug(f"Masked {var}")
    my_logger.info('Masked xy dims')
    # compute max no of samples
    resamp_prd = xarray.DataArray([pd.Timedelta(str(c)).total_seconds() for c in radar.resample_prd.values],
                                  dims='resample_prd', coords=dict(resample_prd=radar.resample_prd))
    max_samples_resamp = radar.time.dt.days_in_month * 24 * 60 * 60 / resamp_prd
    # work out max no of radar samples.
    max_samples = radar.time.dt.days_in_month * 24 * 60 * 60 / radar.sample_resolution.dt.total_seconds()

    # check have no cases with samples > max_samples...
    samples = radar['count_raw_' + variable_name]
    fraction = samples / max_samples
    L = fraction > 1.0
    if L.any():
        ValueError(f"Samples > max_samples for {L.sum()} cases")

    # seasonally group
    radar = radar.resample(time='QS-DEC').map(group_data_set, group_dim='time').compute().load()
    max_samples_resamp = max_samples_resamp.resample(time='QS-DEC').sum().load()  # samples/season
    max_samples = max_samples.resample(time='QS-DEC').sum().load()  # max no of samples/season

    # and select where have 3 months in the dataset.
    if (radar.input_count > 3).any():  # sense check
        ValueError(f"Have {radar.input_count} cases with > 3 months of data")
    L = (radar.input_count == 3).load()
    # and where have at least 70% underlying samples.
    samples = radar['count_raw_' + variable_name]
    fraction = samples / max_samples
    # add in fraction_resample to the radar  dataset.
    radar['fraction'] = fraction
    L = L & (fraction >= 0.7)
    radar = radar.where(L, drop=True).load()
    max_samples_resamp = max_samples_resamp.where(L, drop=True).load()
    max_samples = max_samples.where(L, drop=True).load()
    my_logger.info('resampled')
    # and where count_raw_var_thresh < 20% of samples -- crude mask for artifacts.
    tmsk = radar[f'count_raw_{variable_name}_thresh'] < 0.2 * samples

    # mask out the x-y vars
    for var in xy_vars:
        radar[var] = radar[var].where(tmsk)
        my_logger.debug(f"Masked {var} for max samples")

    # apply masking on resample_prds

    x_y_resample_vars = [v for v in radar.variables if
                         ('x' in radar[v].dims) and ('y' in radar[v].dims) and ('resample_prd' in radar[v].dims)]
    for v in x_y_resample_vars:
        radar[v] = radar[v].where(tmsk, drop=True)
        my_logger.debug(f"Masked {v} for resample_prd")

    # now  write out the data.
    if len(radar.time) == 0:
        raise ValueError("No data left after masking")
    radar = radar.assign_attrs(extra_attrs)
    radar.to_netcdf(out_radar, unlimited_dims='time')
    my_logger.info(f"Written out data to {out_radar}")
