#!/usr/bin/env bash
# Post-process the radar data. First argument is the directory of the radar data to process
# Some other optional args are --region x0 y0 x1 y1 reg_name -- region to extract to and name of region. Coords passed to extract_region.py
# --holdafter -- run first post-processing script after the job(s) have completed.
summary_dir=$1 ; shift
region_name=""
hold_after=""
while (( "$#" )); do
  case "$1" in
    --region)
      if [ -n "$6" ] && [ ${6:0:1} != "-" ]; then
        shift
        region_args="--region $1 $2 $3 $4" ;   shift 4
        region_name="_"$1 ; shift
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --holdafter) # need to specially handle holdafter -- as gets passed through to the first submission
      hold_after=$1; shift
      while (( "$#" )); do
        if [[ $1 == -* ]]; then
          break # Exit the loop if another option is encountered
        fi
        hold_after+=" $1" # Add the argument to hold_after
        shift # Move to the next argument
      done
      ;;
    *)
      extra_args+=" $1" # just add it onto the args for all processing
      shift
      ;;
  esac
done


name=$(basename "$summary_dir")
process_dir=$(realpath ${summary_dir}/../../)"/processed/${name}"
seas_str='DJF'
sm_file="${process_dir}/seas_mean_${name}_${seas_str}.nc"
nomask_file="${process_dir}/seas_mean_${name}_${seas_str}_nomask.nc"
event_file="${process_dir}/events_seas_mean_${name}_${seas_str}${region_name}.nc"
gev_dir="${process_dir}/fits${region_name}"
pbs_log_dir="${process_dir}/pbs_logs"
run_log_dir="${process_dir}/run_logs"
time_str=$(date +"%Y%m%d_%H%M%S")
# run the meaning
job_name="smn_${name}"
log_file=process_seas_avg_mask_${name}_${time_str}
submit_opts=" --submit"
submit_opts+=" --json_submit config_files/process_seas_avg_mask.json --log_base ${pbs_log_dir}/${log_file}"
submit_opts+=" --log_file ${run_log_dir}/${log_file}.log --job_name ${job_name} "
submit_opts+=${hold_after}
cmd="process_seas_avg_mask.py  ${summary_dir} ${sm_file} --no_mask_file ${nomask_file} ${extra_args} ${submit_opts} "
jobid_mean=$($cmd)
status=$?
if [[ $status -ne 0 ]]; then
  echo "Error submitting job $jobid_mean for $cmd" 2>&1
  exit 1
fi
echo "Submitted job $jobid_mean for $cmd" 2>&1
# event processing
job_name="ev_${name}"
log_file=process_events_${name}_${time_str}
submit_opts=" --json_submit config_files/process_events.json --log_base ${pbs_log_dir}/${log_file}"
submit_opts+=" --log_file ${run_log_dir}/${log_file}.log --job_name ${job_name} "
submit_opts+=" --submit --holdafter ${jobid_mean}"
cmd="process_events.py  ${sm_file} ${event_file}  ${extra_args} ${region_args} ${submit_opts}"
jobid_event=$($cmd)
status=$?
if [[ $status -ne 0 ]]; then
  echo "Error submitting job $jobid_event for $cmd" 2>&1
  exit 1
fi
echo "Submitted job $jobid_event for $cmd" 2>&1
# GEV processing
job_name="gev_${name}"
log_file=process_gev_fits_${name}_${time_str}
submit_opts=" --json_submit config_files/process_gev_fits.json --log_base ${pbs_log_dir}/${log_file}"
submit_opts+=" --log_file ${run_log_dir}/${log_file}.log --job_name ${job_name} "
submit_opts+=" --submit --holdafter ${jobid_event}"
cmd="process_gev_fits.py ${event_file} --outdir ${gev_dir} --nsamples=100 --bootstrap=100 ${extra_args} ${submit_opts} "
jobid_gev=$($cmd)
status=$?
if [[ $status -ne 0 ]]; then
  echo "Error submitting job $jobid_gev for $cmd" 2>&1
  exit 1
fi
echo "Submitted job $jobid_gev for $cmd" 2>&1
echo "${jobid_mean}" "${jobid_event}" "${jobid_gev}" # return list of all jobs submitted.

