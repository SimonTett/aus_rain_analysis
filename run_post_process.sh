#!/usr/bin/env bash
# This script is used to generate events data from the radar monthly processed files.
# it generates beam blockage factors and DEM then seasonal avergaes the data for DJF before finally computing the events
#conda activate /home/z3542688/analysis/aus_rain_analysis/mamba_rain_env
#export PYTHONPATH=~/analysis/common_lib:.:$PYTHONPATH
. /home/561/st7295/aus_rain_analysis/setup.sh

# Initialize our own variables
extra_args="" # extra args
name=''
site=''
root_dir=''
process_seas_mean_args=''
seas_str='DJF'
# Loop through all the positional parameters
while (( "$#" )); do
  case "$1" in
    --name)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        name=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --site)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        site=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --root_dir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        root_dir=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --cbb_dir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        cbb_dir=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --years)
      shift
      process_seas_mean_args+=' --years '
      while (( "$#" )) && [[ $1 != -* ]]; do
        process_seas_mean_args+="$1 "
        shift
      done
      ;;
    --season)
      shift
      process_seas_mean_args+=" --season $1"
      seas_str=$1
      shift
      ;;
    --dask)
      extra_args+=" --dask"
      shift
      ;;
    -v)
      extra_args+=" -v"
      shift
      ;;
    --overwrite)
      extra_args+=" --overwrite"
      echo "WARNING: **Overwrite is set**. Sleeping 5 seconds to allow for cancel"
      sleep 5
      shift
      ;;
    --verbose)
      extra_args+=" --verbose"
      shift
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

# Set positional arguments in their proper place
eval set -- "$PARAMS"

# Now you can access your optional argument with $optional_arg and positional arguments with $@

# set up defaults
if [[ -z "$site" ]]; then
    echo "Site is required"
    exit 1
fi
if [[ -z "$name" ]]; then
    echo "Setting name to ${site}"
    name=${site}
fi
if [[ -z "$root_dir" ]]; then
    echo "root_dir is required"
    exit 1
fi
if [[ -z ${cbb_dir} ]]; then
    cbb_dir="/scratch/wq02/st7295/radar/site_data/${site}"
    echo "set cbb_dir to ${cbb_dir}"
fi
input_dir=$1 ; shift
files=${input_dir}/*.nc # get all the files passed as arguments

CBB_dir="${cbb_dir}/${site}"


mn_file="${root_dir}/seas_mean_${name}_${seas_str}.nc"
event_file="${root_dir}/events_${name}_${seas_str}.nc"
fit_root="${root_dir}/gev_fits_${name}_" # for fits -- not currently active.

cmd="./process_beam_blockage.py ${site} --output_dir ${CBB_dir} -v"
# generate the DEM and beam blockage data for the site. Note no extra args used as this only updates rarely.
# User should do it explicitly
echo "cmd is $cmd"
result=$($cmd)
echo "Result is $result"

CBB_files="$CBB_dir"/*cbb_dem.nc # get the beam blockage files
cmds=("./process_seas_avg_mask.py ${files} --output ${mn_file}  ${process_seas_mean_args} --cbb_dem_files ${CBB_files}  ${extra_args}" \
"./process_events.py ${mn_file} ${event_file} --cbb_dem_files ${CBB_files}  ${extra_args}"
#"./process_gev_fits.py ${event_file}  ${fit_root} ${extra_args} # generate the GEV fits for the site."
)
for cmd in "${cmds[@]}"; do
    echo "Running command: $cmd"
    result=$($cmd)
    stat=$?
    echo "Result is $result"
    if [[ $stat -ne 0 ]]; then
        echo "Error running command"
        exit 1
    fi
done

