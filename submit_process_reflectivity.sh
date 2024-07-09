#!/usr/bin/env bash
# submit scripts to process reflectivity data and then (once all ran) submit post-processing
# example useage:
# ./submit_process_reflectivity.sh Melbourne  --calibration melbourne --years 1997 2022
# to_rain values -- Melbourne calibration: 0.0271 0.650
#                    Cape Grim calibration: 0.0224 0.670
#                    Brisbane calibration: 0.0256 0.688
dbz_name=""
year_start=1997
year_end=2022
extra_args=''
calibration='melbourne'
site=$1 ; shift
while (( "$#" )); do
  case "$1" in
    --dbz_range)
      if [ -n "$3" ] && [ ${3:0:1} != "-" ]; then
        shift
        dbz_name="${1}_${2}" # leave args to be be passed on.
        extra_args+=" --dbz_range $1 $2"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --calibration)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        calibration=${2}
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --years)
      if [ -n "$3" ] && [ ${3:0:1} != "-" ]; then
        shift
        year_start=${1}
        year_end=${2}
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi

      ;;
    *)
      extra_args+=" $1"
      shift
      ;;
  esac
done

# Set positional arguments in their proper place



if [[ -z "$site" || -z "$calibration" ]]
then
    echo "Usage: $0 site calibration [extra_args]"
    exit 1
fi
if [[ -z "$AUSRAIN" ]]
then
    echo "AUSRAIN environment variable not set. Please set it to the aus_rain_analysis directory."
    exit 1
fi

resample='30min 1h 2h 4h 8h'
region="-125 125 125 -125" # region to extract
coarse=4

extra_args+=" --dask --min_fract_avg 0.75 --threshold 1.0 --extract_coords_csv meta_data/${site}_close.csv"
extra_args+=" --json_submit_file ${AUSRAIN}/config_files/process_reflectivity.json --submit"
if [[ $calibration == "melbourne" ]]
then
    extra_args+=" --to_rain 0.0271 0.650"
elif [[ $calibration == "Grim" ]]
then
    extra_args+=" --to_rain 0.0224 0.670"
elif [[ $calibration == "brisbane" ]]
then
    extra_args+=" --to_rain 0.0256 0.688"
else
    echo "Unknown calibration $calibration"
    exit 1
fi
if [[ -n "$resample" ]]
then
    extra_args+=" --resample ${resample}"
fi
if [[ -n "$region" ]]
then
    extra_args+=" --region ${region}"
fi
if [[ -n "$coarse" ]]
then
    extra_args+=" --coarsen ${coarse} ${coarse}"
fi
out_name=${site}_rain_${calibration}_${dbz_name}
time_str=$(date +"%Y%m%d_%H%M%S")
root_dir="/scratch/wq02/st7295/radar/summary"
out_dir="${root_dir}/${out_name}"
all_jobs=''
for year in $(seq ${year_start} ${year_end}); do # parallel jobs -- one for each year
  run_log_file="${out_dir}/run_logs/${out_name}_${year}_${time_str}.log"
  pbs_log_file="${out_dir}/pbs_logs/${out_name}_${year}_${time_str}"
  job_name=${site}_${year}_ref_${calibration}_${dbz_name}
  cmd="process_reflectivity.py ${site} ${root_dir}/$out_name $extra_args --job_name ${job_name} --years ${year} $*"
  cmd+=" --log_file ${run_log_file} --log_base ${pbs_log_file}"
  job_id=$($cmd)
  stat=$?
  if [[ $stat -ne 0 ]]
  then
      echo "Error processing reflectivity for ${site} ${year} ${calibration} ${dbz_name}"
      exit 1
  else
      echo "Submitted job ${job_name} as ${job_id}" >&2
  fi
  all_jobs+=":${job_id}"
done
all_jobs=${all_jobs:1} # remove the leading colon
echo "Submitted all max jobs. " >&2
echo ${all_jobs} # return the last jobid. Allows pp to be submitted with a hold on this job.


