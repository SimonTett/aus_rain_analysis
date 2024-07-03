#!/usr/bin/env bash
# Post-process the radar data. First argument is the directory of the radar data to process
summary_dir=$1 ; shift
name=$(basename "$summary_dir")
process_dir=$(realpath ${summary_dir}/../../)"/processed/${name}"
seas_str='DJF'
sm_file="${process_dir}/seas_mean_${name}_${seas_str}.nc"
nomask_file="${process_dir}/seas_mean_${name}_${seas_str}_nomask.nc"
event_file="${process_dir}/events_seas_mean_${name}_${seas_str}.nc"
gev_dir="${process_dir}/gev_fits"
pbs_log_dir="${process_dir}/pbs_logs"
run_log_dir="${process_dir}/run_logs"
time_str=$(date +"%Y%m%d_%H%M%S")
# run the meaning
job_name="smn_${name}"
log_file=process_seas_avg_mask_${name}_${time_str}
submit_opts=" --json_submit config_files/process_seas_avg_mask.json --log_base ${pbs_log_dir}/${log_file}"
submit_opts+=" --log_file ${run_log_dir}/${log_file}.log --job_name ${job_name} "
submit_opts+=" --submit"
cmd="process_seas_avg_mask.py  ${summary_dir} ${sm_file} --no_mask_file ${nomask_file} ${submit_opts} $*"
jobid=$($cmd)
status=$?
if [[ $status -ne 0 ]]; then
  echo "Error submitting job $jobid for $cmd" 2>&1
  exit 1
fi
echo "Submitted job $jobid for $cmd" 2>&1
# event processing
job_name="ev_${name}"
log_file=process_events_${name}_${time_str}
submit_opts=" --json_submit config_files/process_events.json --log_base ${pbs_log_dir}/${log_file}"
submit_opts+=" --log_file ${run_log_dir}/${log_file}.log --job_name ${job_name} "
submit_opts+=" --submit --holdafter ${jobid}"
cmd="process_events.py  ${sm_file} ${event_file} ${submit_opts} $*"
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
cmd="process_gev_fits.py ${event_file} --outdir ${gev_dir} --nsamples=100 --bootstrap=100 ${submit_opts} $* "
jobid_gev=$($cmd)
status=$?
if [[ $status -ne 0 ]]; then
  echo "Error submitting job $jobid_gev for $cmd" 2>&1
  exit 1
fi
echo "Submitted job $jobid_gev for $cmd" 2>&1

