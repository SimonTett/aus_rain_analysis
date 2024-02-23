#!/bin/env python
import argparse
import glob
import sys
import os
import itertools
import multiprocessing

import xarray

import ausLib
import pathlib
import logging

if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!
    parser = argparse.ArgumentParser(prog="comp_radar_max", description="Compute the max of time-average means")
    parser.add_argument("INPUTS", nargs='+', help='list of files to process', type=str)
    parser.add_argument("--mean_resample", help='resample values to use for mean', type=str, default="1h")
    parser.add_argument("--max_resample", "-r", help='resample values to use for max', type=str, default="MS")
    parser.add_argument("OUTPUT", help="Name of output file")
    parser.add_argument("--coordinate_names",
                        help='Names of coordinates in order x, y, time',
                        nargs=3, default=['x', 'y', 'time'])
    parser.add_argument('--overwrite', action='store_true', help='If set then overwrite existing output')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Be verbose. Once = info, twice debug, none warning')
    parser.add_argument("--dask", action='store_true', help="If set will start dask.distributed Client.")
    parser.add_argument("--log_file", help='Name of logging file for log output. ')
    parser.add_argument("--year_chunk", action='store_true',
                        help="""Process each calendar year separately based on file name. This runs considerably 
                        faster than reading in all data and processing in one go but means no seasonal processing.""")
    parser.add_argument("--cache", action="store_true",
                        help="""When running with --year_chunk save each year. If cached files exist they will be loaded.""")
    parser.add_argument("--keep_cache",action="store_true",help='Keep cached files at end of processing')
    args = parser.parse_args()
    if args.verbose >= 2:
        level = 'DEBUG'
    elif args.verbose == 1:
        level = 'INFO'
    else:
        level = 'WARNING'
    mode = 'a'
    if args.overwrite:
        mode = 'w'
    log = logging.getLogger(ausLib.__name__)

    ausLib.init_log(log, level, log_file=args.log_file, mode=mode)

    output = pathlib.Path(args.OUTPUT)
    # run glob on everything
    inputs = []
    for f in args.INPUTS:
        inputs += glob.glob(f)
    # check have some files -- if not exit
    if len(inputs) == 0:
        raise FileNotFoundError(f"No files found from expanding {args.INPUTS}")
    # and then convert to paths
    inputs = [pathlib.Path(f) for f in sorted(inputs)]

    # check all files exist. Complain if not and die. While doing this work out the year to make chunking easier.
    # seems to be needed to run on gadi
    files_ok = True
    years = dict()
    for f in inputs:
        if not f.exists():
            log.warning(f"File {f} does not exist")
            files_ok = False
        year = int(f.name.split("_")[1][0:4])
        years[year] = years.get(year, []) + [f]
    if not files_ok:
        raise FileExistsError("Some files do not exist")

    # chunks = dict(zip(args.coordinate_names[0:2], [51, 51, 72]))  # guess for good chunking values.
    chunks = {}  # default
    if args.dask:
        client = ausLib.dask_client()

    if args.year_chunk:
        # shoudl probably store the processing options to properly cache...
        output_files=[] # where we are going to put the annual chunks.
        year_values = sorted(list(years.keys()))
        result_list = []
        for year in year_values:
            if args.cache:
                output_file = output.parent / "cache"/(output.stem + f"_cache_{year}.nc")
                output_files.append(output_file)
            else:
                output_file=None
            if args.cache and output_file and output_file.exists():
                log.info(f"Loading existing file {output_file}")
                cached_result = xarray.load_dataset(output_file)
                result_list.append(cached_result)
            else:
                input_files = years[year]
                files_missing = 0
                for f in input_files:
                    if not f.exists():
                        log.warning(f"File {f} does not exist. ")
                        files_missing += 1
                if files_missing > 0:
                    raise  FileNotFoundError(f"{files_missing} files do not exist")
                log.info(f"Processing {len(input_files)} files for year {year} and writing to {output_file}")
                result_list += [ausLib.summary_radar(input_files, output_file,
                                                     mean_resample=args.mean_resample,
                                                     summary_resample=args.max_resample,
                                                     chunks=chunks,
                                                     overwrite=args.overwrite,
                                                     time_dim=args.coordinate_names[2])]

        # end of processing
        log.info("Concatenating all years")
        result = xarray.concat(result_list, dim=args.coordinate_names[2])
        log.info("Writing data  out")
        # make dir if needed
        output.parent.mkdir(parents=True,exist_ok=True)
        result.to_netcdf(output, format='NETCDF4')
        log.info(f"Wrote data to {output}")
        # remove all the output files
        if not args.keep_cache:
            for f in output_files:
                f.unlink()
                log.debug(f"Deleted {f}")
            if len(output_files):
                log.info(f"Removed {len(output_files)} year_chunk cached output files")
    else:
        result = ausLib.summary_radar(inputs, output,
                                      mean_resample=args.mean_resample,
                                      summary_resample=args.max_resample,
                                      chunks=chunks,
                                      overwrite=args.overwrite,
                                      time_dim=args.coordinate_names[2])
