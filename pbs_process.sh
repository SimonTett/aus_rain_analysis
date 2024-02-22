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
# Melbourne takes  about 4 minutes  to process a year.

export name=adelaide # name of place. Lowercase and in lookup table.
export year=2020 # year to process. Do not set if want to process everything
echo "Name is $name"
cd /home/561/st7295/aus_rain_analysis || exit # make sure we are in the right directory
. ./setup.sh # setup software and then run the processing

declare -A radar_numbers # translation from names to numbers for the radars
radar_numbers=([adelaide]=46 [melbourne]=2 [wtakone]=52 [sydney]=3 [brisbane]=50 [canberra]=40 \
  [cairns]=19 [mornington]=36 [grafton]=28 [newcastle]=4 [gladstone]=23)
number=${radar_numbers[${name}]}

if [[ -z "${number}" ]]
then
  echo "failed to find ${name} in radar_numbers table. Exiting"
  exit 1
fi

pattern="${number}_*rainrate.nc"
output="radar/${name}/processed_${name}.nc"
log_file="radar/${name}/processed_${name}.log"
extra="--year_chunk"
if [[ -n "${year}" ]]
then
  pattern="${number}_${year}*rainrate.nc"
  output="radar/${name}/processed_${year}_${name}.nc"
  log_file="radar/${name}/processed_${year}_${name}.log"
  extra=""
fi
cmd="./comp_radar_max.py /g/data/rq0/level_2/${number}/RAINRATE/${pattern} ${output} \
 --overwrite --verbose --dask --log_file ${log_file} ${extra}"
 echo "cmd is: ${cmd}"
 result=$($cmd)  # executing cmd
 echo $result

