#!/usr/bin/env bash
# submit  all gev analysis, All args are passed straight through to the individual submit_post_process.sh scripts
# Loop through all the positional parameters
while (( "$#" )); do
  case "$1" in
    --single)
      shift # run one job at a time
      single=True
      ;;
    *) # everything else gets puts into params
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done
# reset args
eval set -- "$PARAMS"
time_str=$(date +"%Y%m%d_%H%M%S")

files=$(ls -1 /scratch//wq02/st7295/radar/processed/*/events*.nc ) # all event files
if [[ -n $single ]]; then
  echo "Single not implemented"
  exit 1
fi
for file in $files; do
  job_name=$(basename $file .nc)
  job_name=${job_name/events_/gev_}
  pbs_log_file="/scratch/wq02/st7295/radar/processed/pbs_logs/${job_name}_${time_str}"
  run_log_file="/scratch/wq02/st7295/radar/processed/run_logs/${job_name}_${time_str}.log"
  cmd="process_gev_fits.py $file --log_file ${run_log_file} $* \
  --submit --json_submit config_files/process_gev_fits.json \
  --job_name $job_name --log_base $pbs_log_file "
  #echo $cmd
  jobid=$($cmd)
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "Error submitting job $jobid for $cmd" 2>&1
    exit 1
  fi
  echo "submitted job $jobid for $cmd" 2>&1
done
echo "All jobs submitted" 2>&1
echo $jobid
