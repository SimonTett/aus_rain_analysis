#!/usr/bin/env bash
# This script is used to generate events data from the radar monthly processed files.
# it generates beam blockage factors and DEM then seasonal avergaes the data for DJF before finally computing the events
#conda activate /home/z3542688/analysis/aus_rain_analysis/mamba_rain_env
#export PYTHONPATH=~/analysis/common_lib:.:$PYTHONPATH
. /home/561/st7295/aus_rain_analysis/setup.sh
base_dir="/scratch/wq02/st7295/radar"
# Initialize our own variables
extra_args="" # extra args
name=''
root_dir=''
process_seas_mean_args=''
seas_str='DJF'
# Loop through all the positional parameters
while (( "$#" )); do
  case "$1" in
    --input_dir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        input_dir=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --name)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        name=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --outdir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        outdir=$2
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
    --dryrun)
      dryrun='True'
      echo "Dry run only"
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
site=$1 ; shift
if [[ -z "$site" ]]; then
    echo "Site is required"
    exit 1
fi
if [[ -z "$name" ]]; then
    echo "Setting name to ${site}" >&2
    name=${site}
fi
if [[ -z "$input_dir" ]]; then
    input_dir="${base_dir}/summary/${name}"
    echo "setting input_dir to ${input_dir}" >&2
fi
if [[ -z "$outdir" ]]; then
    outdir="${base_dir}/processed/${name}"
     echo "setting outdir to ${outdir}" >&2
fi
if [[ -z ${cbb_dir} ]]; then
    cbb_dir="${base_dir}/site_data/${site}"
    echo "set cbb_dir to ${cbb_dir}" >&2
fi

if [[ -z ${input_dir} ]]; then
    input_dir="${base_dir}/summary/${name}"
    echo "set input_dir to ${input_dir}" >&2
fi

files="${input_dir}/*.nc" # get all the files passed as arguments
mn_file="${outdir}/seas_mean_${name}_${seas_str}.nc"
event_file="${outdir}/events_${name}_${seas_str}.nc"


CBB_files="$cbb_dir/${site}*cbb_dem.nc" # get the beam blockage files
cmds=("./process_seas_avg_mask.py ${files} --output ${mn_file}  ${process_seas_mean_args} --cbb_dem_files ${CBB_files}  ${extra_args}" \
"./process_events.py ${mn_file} ${event_file} --cbb_dem_files ${CBB_files}  ${extra_args}"

)
for cmd in "${cmds[@]}"; do
    # shellcheck disable=SC2028
    if [[ -n "${dryrun}" ]]; then # dryrun ??
      echo -e 'cmd is:\n '"${cmd}"
      continue
    fi
    result=$($cmd)
    stat=$?
    echo -e "$cmd:\n Result is $result"
    if [[ $stat -ne 0 ]]; then
        echo "Error running command $cmd"
        exit 1
    fi
done

