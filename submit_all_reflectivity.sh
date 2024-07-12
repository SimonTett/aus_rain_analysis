#!/usr/bin/env bash
# submit  all reflectivity calcs.  All args are passed straight through to the individual submit_process_reflectivity.sh scripts
# edit the script to change the rain values and rain_name
# example use is ./submit_all_reflectivity.sh  --min_fract_avg 0.75 --threshold 1.0 -v
#TODO update.  This script is out of date.
args=$*
to_rain_args='--to_rain 0.0271 0.650' # melbourne values
rain_name='_rain_melbourne'
to_rain_args='--to_rain 0.0256 0.688' # Brisbane values
rain_name='_rain_brisbane'
echo "submitting reflectivity for $rain_name with args: $args"
echo "sleeping 5 seconds. "
sleep 5

sites="Adelaide Melbourne Wtakone Sydney Brisbane Canberra Cairns Mornington Grafton Newcastle Gladstone"
#sites="Adelaide Melbourne" # for testing
holdafter=''
for site in ${sites}; do
  echo "submitting reflectivity for $site"
  cmd="./submit_process_reflectivity.sh ${site} --name "${site}${rain_name}" ${to_rain_args} --extract_coords_csv meta_data/${site}_close.csv ${holdafter} $args"
  echo "cmd is ${cmd}"
  job=$($cmd) 2>&1 # put everthing on std out
  holdafter="--holdafter ${job}"
  echo "job: ${job}"
  echo "---------------------------"
done
