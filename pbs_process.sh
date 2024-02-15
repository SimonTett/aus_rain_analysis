#!/usr/bin/env bash
# run the shell scrip to process a bunch of data. About 1.4 minutes/year on 8 cores. Parallel speedup about 4.
#PBS -P wq02
#PBS -q normalbw
#PBS -l walltime=02:00:00
#PBS -l storage=gdata/rq0+gdata/ua8+gdata/hh5
#PBS -l mem=60GB
#PBS -l ncpus=8
#PBS -l jobfs=2GB
#PBS -l wd
#PBS -m abe
#PBS -M simon.tett@ed.ac.uk
#PBS -N process_radar
#PBS -o /home/561/st7295/aus_rain_analysis/sydney/pbs_output
#PBS -e /home/561/st7295/aus_rain_analysis/sydney/pbs_output
# Takes 32 minutes to run 24 years for Grafton data. Needs a lot of Memory so might as well use CPUs to run in parallel.
# While Sydney takes about 2 minute/year to process
cd /home/561/st7295/aus_rain_analysis || exit # make sure we are in the right directory
. ./setup.sh # setup software and then run the processing
name='port_hedland' # name of radar
number=16 # number of radar
cmd="./comp_radar_max.py /g/data/rq0/level_2/${number}/RAINRATE/${number}_*rainrate.nc radar/${name}/processed_radar_${name}.nc\
 --overwrite --verbose --dask --log_file radar/${name}/processed_radar_${name}.log --year_chunk"
 echo "cmd is: ${cmd}"
 $($cmd)  # executing cmd

