#!/bin/env python
import argparse
import ausLib
import pathlib
import logging
parser = argparse.ArgumentParser(prog="comp_rolling_max",description="Compute the max of the rolling mins")
parser.add_argument("INPUTS",nargs='+',help='list of files to process',type=str)
parser.add_argument("--resample","-r",help='resample values to use',type=str,default="1h")
parser.add_argument("--variables",nargs='+',help='Variable names to use',type=str,default=None)
parser.add_argument("OUTPUT",help="Name of output file")
parser.add_argument("--coordinate_names",
                    help='Names of coordinates in order x, y, time',
                    nargs=3,default=['x','y','time'])
parser.add_argument('--overwrite',action='store_true',help='If set then overwrite existing output')

parser.add_argument('--verbose','-v',action='count',default=0,
                    help='Be verbose. Once = info, twice debug, none warning')

args=parser.parse_args()
if args.verbose >= 2:
    level='DEBUG'
elif args.verbose == 1:
    level='INFO'
else:
    level='WARNING'

logging.basicConfig(force=True,level=level,
                    format='%(asctime)s %(levelname)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
output = pathlib.Path(args.OUTPUT)
inputs = [pathlib.Path(f) for f in args.INPUTS]

chunks=dict(zip(args.coordinate_names,[20,20,144])) # guess for good chunking values.
chunks={} # default
result = ausLib.seasonal_max(inputs,output,variables=args.variables,
                                      resample=args.resample,
                                      chunks=chunks,overwrite=args.overwrite,
                                      time_dim=args.coordinate_names[2])

