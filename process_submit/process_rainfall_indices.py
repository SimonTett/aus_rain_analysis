# %run -i process_submit/process_rainfall_indices --glob "09" --months 1 --indir "C:\Users\stett2\OneDrive - University of Edinburgh\data\aus_radar_analysis\radar\raw_radar_data" Adelaide data
# test code to produce equivalent QC to Bowden et al.
# However rather than QCing "events" will QC days.
"""
Process
In the process of analyzing the event data sets from each radar, 565 events were found that were primarily
comprised of ground clutter or anomalous propagation. Examination of the characteristics of these events
revealed that they typically had low areal coverage and high stratiform intensities. In addition to removing the 565
events identified manually, any event with a value of (Rain Area Fraction × Convective Area Fraction)/Stratiform
Intensity below the 75th percentile of the manually identified false events was omitted from analysis. This left a
final data set of 23,748 rainfall events from the combined 15 radars for analysis.

Terms:
rain area fraction = number of pixels with non-zero rain divided by the total number of pixels in the scene.
Convective Area Fraction = number of convective pixels with non-zero rain divided by the total number of pixels in the scene.
Stratiform intensity = average rain rate over stratiform pixels with non-zero rain rates only.

Can compute these for each radar slice and compute timeseries. These get saved when done.
Hard part matching filenames.
"""
import pathlib
import typing
import xarray
import numpy as np
import argparse
import multiprocessing
import pandas as pd

import ausLib


def steiner_path(reflect_path: pathlib.Path,
                 base_reflect:pathlib.Path =ausLib.hist_ref_dir,
                 base_level_2:pathlib.Path = ausLib.radar_dir) -> pathlib.Path :
    """
    Work out path to steiner classification file for input reflectivity data.
    Args:
       reflect_path: Path to reflectivity data.
       base_reflect: Path to base directory for reflectivity data.
       base_level_2: Path to base directory for steiner classification data.

    Returns: path to steiner classification file for input reflectivity data.

    """

        # on gadi base is /g/data/rq0/level_2
    # Reflectivity paths are of form base_reflect/$SITE/year/$SITE_YYYYMMDD.gndrefl.zip
    # Steiner classification parts are form base_level_2/$SITE/STEINER/$SITE_YYYYMMDD_steiner.nc

    steiner_path = base_level_2 / reflect_path.relative_to(base_reflect)
    # now to remove the year cpt and change .gndrefl.zip to _steiner.nc in the name
    steiner_path = steiner_path.parent.parent/ 'STEINER' / (steiner_path.stem.split('.')[0] + '_steiner.nc')
    if not steiner_path.exists():
        raise FileNotFoundError(f"{steiner_path} does not exist")
    return steiner_path

def comp_bowden_indices(radar_file:pathlib.Path,
    to_rain:typing.Optional[tuple[float,float]] = None,
    dbz_ref_limits:typing.Optional[tuple[float,float]] = None)  -> xarray.Dataset:
    """"
        Compute indices used in Bowden et al, 2025:  https://doi.org/10.1029/2024JD041790
        :param radar_file - file to radar data
        :param to_rain -- tuple of to_rain values pased to read_zip.
           Default is Melbourne values of  0.0271 & 0.65
        :param dbz_ref_limits -- dbz limits. Passed to read_zip.
          Default is 15.0 &55.
        """
    # defaults
    if to_rain is None:
        to_rain = (0.0271, 0.650) # Melbourne distrometer values
    if dbz_ref_limits is None:
        dbz_ref_limits = (15.,55.)
    spatial_dims = ['x', 'y']
    drop_vars_first = ['error', 'reflectivity', 'doppler_velocity',
                       'count_rain_rate_missing']  # variables not to read for meta info
    drop_vars = ['error', 'x_bounds', 'y_bounds', 'proj', 'doppler_velocity',
                 'count_rain_rate_missing']  # variables not to read for data
    st = steiner_path(radar_file)
    rain = ausLib.read_zip(radar_file, to_rain=to_rain, dbz_ref_limits=dbz_ref_limits, coarsen=dict(x=2, y=2),
                    region=dict(x=slice(-126.5, 126.5), y=slice(126.5, -126.5))
                           )  # compat with steiner calculation.
    rain = rain.rain_rate
    my_logger.debug(f"Reading steiner data from {st}")
    ## read steiner and do various fixes.
    steiner = xarray.load_dataset(st)  # read in steiner classification.
    # extract what we want and change time name
    steiner = steiner.steiner.sel(time=rain.valid_time.values, method='nearest', tolerance=np.timedelta64(60, 's'))
    # fix times
    steiner = steiner.rename(dict(time='valid_time'))
    steiner.coords['valid_time'] = rain.valid_time

    # fix the co-ords
    for c in ['x', 'y']:
        cv = rain.coords[c]
        steiner.coords[c] = steiner.coords[c] / 1000.  # convert to km
        steiner = steiner.sel({c: slice(cv.min(), cv.max())})

    rain = xarray.where(steiner.notnull(), rain, steiner)  # mask rain by steiner
    ## now to compute the indices that Bowden et al computed.
    total_pixels = rain.notnull().sum(spatial_dims)  # number of non-missing pixels
    raining = rain.where(rain > 0)  # where we have some rain.
    # fraction of region raining/convecting or stratiform
    rain_fraction = raining.notnull().sum(spatial_dims) / total_pixels  # fraction raining pixels
    conv_rain_fraction = raining.where(steiner == 2).notnull().sum(spatial_dims) / total_pixels
    strat_rain_fraction = raining.where(steiner == 1).notnull().sum(spatial_dims) / total_pixels
    # Intensity
    strat_intensity = raining.where(steiner == 1).mean(spatial_dims)  # intensity of strat rain
    conv_intensity = raining.where(steiner == 2).mean(spatial_dims)  # intensity of conv rain
    rain_intensity = raining.mean(spatial_dims)
    # stick them all in a dataset
    ds = xarray.Dataset(
        dict(rain_fraction=rain_fraction,
             conv_fraction=conv_rain_fraction,
             strat_fraction=strat_rain_fraction,
             rain_intensity=rain_intensity,
             conv_intensity=conv_intensity,
             strat_intensity=strat_intensity))

    # add metadata?
    return ds

def multi_comp_bowden_indices(radar_files: list[pathlib.Path],
                        to_rain:typing.Optional[tuple[float,float]] = None,
                        dbz_ref_limits:typing.Optional[tuple[float,float]] = None) -> xarray.Dataset:
    """"
    Compute indices used in Bowden et al, 2025:  https://doi.org/10.1029/2024JD041790
    :param radar_files - list of radar files
    :param to_rain -- tuple of to_rain values pased to read_zip.
       Default is Melbourne values of  0.0271 & 0.65
    :param dbz_ref_limits -- dbz limits. Passed to read_zip.
      Default is 15.0 &55.
    :spatial dims -- list of spatial dims. Default is ['x','y']
    """



    ds_list = []
    for radar_file in radar_files:
        ds = comp_bowden_indices(radar_file, to_rain=to_rain, dbz_ref_limits=dbz_ref_limits)
        ds_list.append(ds)
        my_logger.info(f"Done with {radar_file}. Memory {ausLib.memory_use()}")

    my_logger.info(f'Read in {len(ds_list)} files {ausLib.memory_use()}')
    ds_bowden = xarray.concat(ds_list,dim='valid_time').rename(dict(valid_time='time')).sortby('time')
    return ds_bowden


def bowden_reality_factor(ds: xarray.Dataset) -> xarray.DataArray:
    """
    Compute reality factor from Bowden et al 2025
    Args:
        ds: xarray dataset containing rain_fraction, conv_fraction and strat_intensity

    Returns:

    """


    # when strat_intensity is zero we end up with infinite reality factor.
    # But if we have NO convective rain then have conv_rain_fraction =0  and that will give 0 reality factor.
    # What do then? Same when rain_fraction is 0. I *think* that will have 0 strat intensity though

    # mask out fractions where intensity = nan.
    conv_fraction = ds.conv_fraction.where(ds.conv_intensity.notnull())
    rain_fraction= ds.rain_fraction.where(ds.rain_intensity.notnull())
    reality_factor = xarray.where(ds.strat_intensity > 0,
                                  rain_fraction*conv_fraction/ds.strat_intensity,np.nan)
    return reality_factor

if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!

    parser = argparse.ArgumentParser(description='Process reflectivity data')

    parser.add_argument('site', help='Radar site to process',
                        default='Melbourne', choices=ausLib.site_numbers.keys())
    parser.add_argument('outdir', type=pathlib.Path,
                        help='output directory. Will be created if it does not exist. If not set will be cwd()/site',
                        nargs='?')
    parser.add_argument('--indir', type=pathlib.Path, help='input directory',
                        default = ausLib.hist_ref_dir)
    parser.add_argument('--glob', help='Pattern for globbing zip files',
                        default='[0-9][0-9].gndrefl.zip')
    parser.add_argument('--years', nargs='+', type=int, help='List of years to process',
                        default=range(2020, 2023))
    parser.add_argument('--months', nargs='+', type=int,
                        help='list of months to process', default=range(1, 13))


    parser.add_argument('--dbz_range', nargs=2, type=float, default=[15., 55.],
                        help='range for dbz ref. '
                             'Values below are set to 0 when converting DBZ to linear units; above to missing')
    parser.add_argument('--to_rain', type=float, nargs=2,
                        help='Convert Reflectivity to rain using R=c[0]Z^c[1]. Default are Melbourne dist values',
                        default=[0.0271,0.65])
    #parser.add_argument('--region', nargs=4, type=float, help='Region to extract data for as x0 x1 y0 y1')

    ausLib.add_std_arguments(parser)
    args = parser.parse_args()
    my_logger = ausLib.process_std_arguments(args)  # deal with the std arguments



    # print out all the arguments and add them to attributes of the final dataset.

    extra_attrs = dict(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.utcnow().isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()],
                       site=args.site, dbz_range=args.dbz_range,to_rain=args.to_rain)


    site_number = f'{ausLib.site_numbers[args.site]:d}'
    # deal with some args.


    indir = args.indir # if use rainfall3 then this varies.
    glob = args.glob # if use rainfall3 then this varies.


    indir = indir/site_number # work out directory for input data. Depends on site_number.
    my_logger.info(f'Input directory is {indir}')
    if not indir.exists():
        my_logger.warning('Input directory {indir} does not exist')
        raise FileNotFoundError(f'Input directory {indir} does not exist')

    outdir = args.outdir
    if outdir is None:
        outdir = pathlib.Path.cwd()/args.site
    outdir.mkdir(parents=True, exist_ok=True)
    my_logger.info(f'Output directory is {outdir}')
    ## dealt with arguments. Now to do the processing.
    for year in args.years:
        my_logger.info(f'Processing year {year}')
        for month in args.months:
            data_dir = indir / f'{year:04d}'
            pattern = f'{site_number}_{year:04d}{month:02d}' + glob
            file = f'bowden_indices{year:04d}_{month:02d}.nc' # name of output file
            zip_files = sorted(list(data_dir.glob(pattern)))
            if len(zip_files) == 0:
                my_logger.info(f'No files found for   {data_dir}.glob{pattern} in {data_dir} {ausLib.memory_use()}')
                continue
            my_logger.info(f'Found {len(zip_files)} files for {data_dir}{pattern} {ausLib.memory_use()} ')
            outpath = outdir / file
            if (not args.overwrite) and outpath.exists():
                my_logger.warning(f'{outpath} exists  skipping processing. Use --overwrite')
                continue
            bowden_indices = multi_comp_bowden_indices(zip_files,
                                                       dbz_ref_limits=args.dbz_range,
                                                       to_rain=args.to_rain)

            # write the data out.
            ausLib.write_out(bowden_indices, outpath,  extra_attrs=extra_attrs)




