#!/usr/bin/env bash
# submit seasonal mean processing. First argument is the directory of the radar data to process
# all other arguments are passed through to process_seas_avg_mask.py
# Need to have run setup.sh before running this script.
# log files etc will go to the radar processed directory.
# example of how to run this to generate all the monthly files.
# for f in $(ls -1d /scratch/wq02/st7295/radar/summary/* | grep -v coord )
# do echo $f
# submit_seas_mean.sh  $f --season monthly -v -v
# done

if [[ -z $AUSRAIN ]]; then
  echo "Error: Must define AUSRAIN first. Have you done . setup.sh ??" >&2
  exit 1
fi
summary_dir=$1; shift
name=$(basename "$summary_dir")
process_dir=$(realpath "${summary_dir}"/../../)"/processed/${name}"
pbs_log_dir="${process_dir}/pbs_logs"
run_log_dir="${process_dir}/run_logs"
time_str=$(date +"%Y%m%d_%H%M%S")
job_name="smn_${name}"
log_file=process_seas_avg_mask_${name}_${time_str}
submit_opts=" --submit"
submit_opts+=" --json_submit ${AUSRAIN}/config_files/process_seas_avg_mask.json --log_base ${pbs_log_dir}/${log_file}"
submit_opts+=" --log_file ${run_log_dir}/${log_file}.log --job_name ${job_name} "
cmd="process_seas_avg_mask.py  ${summary_dir} $* ${submit_opts} "
jobid_mean=$($cmd)
status=$?
if [[ $status -ne 0 ]]; then
  echo "Error submitting job $jobid_mean for $cmd" 2>&1
  exit 1
fi
echo "Submitted job $jobid_mean for $cmd" 2>&1
echo $jobid