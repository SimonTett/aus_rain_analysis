#!/usr/bin/env python
# extract meta data from PPI files.
import xradar # this should happen early so that radar magic can happen!
import xarray
import itertools
import multiprocessing
import argparse
import ausLib
import pathlib
import pandas as pd
import dask

my_logger = ausLib.my_logger

def month_sort(path:pathlib.Path) -> str:
    """
    Sort fn to give yyyy-mm order.
    Args:
        path (): pathlib.path

    Returns: yyyy-mm


    """
    yyyymm = path.name.split('_')[1][0:6]
    return yyyymm

if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!

    parser = argparse.ArgumentParser(description='Extract data from PPI files')
    parser.add_argument('site', help='Radar site to process',
                        default='Melbourne', choices=ausLib.site_numbers.keys())
    parser.add_argument('--years', nargs='+', type=int, help='List of years to process',
                        default=range(1990, None))
    parser.add_argument('outfile', type=pathlib.Path,help='output file for metadata. Must be provided')
    ausLib.add_std_arguments(parser) # add std args
    args = parser.parse_args()
    time_unit = 'minutes since 1970-01-01'  # units for time in output files
    # deal with verbosity!
    if args.verbose > 1:
        level = 'DEBUG'
    elif args.verbose > 0:
        level = 'INFO'
    else:
        level = 'WARNING'
    ausLib.init_log(my_logger, level=level, log_file=args.log_file, mode='w')
    extra_attrs = dict(program_name=str(pathlib.Path(__file__).name),
                       utc_time=pd.Timestamp.utcnow().isoformat(),
                       program_args=[f'{k}: {v}' for k, v in vars(args).items()],
                       site=args.site)
    for name, value in vars(args).items():
        my_logger.info(f"Arg:{name} =  {value}")
    if args.dask:
        my_logger.info('Starting dask client')
        client = ausLib.dask_client()
    else:
        dask.config.set(scheduler="single-threaded")  # make sure dask is single threaded.
        my_logger.info('Running single threaded')

    site_number = f'{ausLib.site_numbers[args.site]:d}'
    indir = pathlib.Path('/g/data/rq0/level_1b/2/ppi/')/ site_number
    my_logger.info(f'Input directory is {indir}')
    if not indir.exists():
        raise FileNotFoundError(f'Input directory {indir} does not exist')
    outfile = args.outfile
    outfile.parent.mkdir(parents=True, exist_ok=True)
    my_logger.info(f'Output directory is {outfile}')
    for year in args.years:
        my_logger.info(f'Processing year {year}')
        pattern = f'{site_number}_{year:04d}*_ppi.zip'
        data_dir = indir / f'{year:04d}'
        zip_files = sorted(data_dir.glob(pattern), key=month_sort)
        if len(zip_files) == 0:
            my_logger.info(f'No files found for  pattern {pattern} in {data_dir} {ausLib.memory_use()}')
            continue
        my_logger.info(f'Found {len(zip_files)} files for pattern {pattern} {ausLib.memory_use()} ')
        # these are all daily files. But we want the first one from each month.
        files=[]
        ukeys=[]
        for k,g in itertools.groupby(zip_files, key=month_sort):
            files.append(list(g)[0])
            ukeys.append(k)
        dataset=[]
        drop_var=['corrected_reflectivity','path_integrated_attenuation','time_coverage_start','time_coverage_end']
        for f in files:
            ds_first = ausLib.read_radar_zipfile(f,first_file=True,
                                                 drop_variables=drop_var) # just extract metadata from the first file.
            dataset+= [ds_first]
        result = xarray.concat(dataset, dim='time')
        # write out the data???

