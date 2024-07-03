#!/usr/bin/env bash
# submit scripts to compute beam blockage factors. Provide the site.
# Other options passed through to process_beam_blockage.py
time_str=$(date +"%Y%m%d_%H%M%S")
site=$1 ; shift
if [[ -z "$site" ]]
then
    echo "Error: must provide site as first argument" >&2
    exit 1
fi
process_dir="/scratch/wq02/st7295/radar/site_data/${site}"
pbs_log_dir="${process_dir}/pbs_logs"
run_log_dir="${process_dir}/run_logs"
job_name="block_${site}"
log_file=block_${site}_${time_str}
submit_opts=" --json_submit config_files/process_beam_blockage.json --log_base ${pbs_log_dir}/${log_file}"
submit_opts+=" --log_file ${run_log_dir}/${log_file}.log --job_name ${job_name} "
submit_opts+=" --submit"
cmd="./process_beam_blockage.py ${site} ${*} ${submit_opts}"
jobid=$($cmd)
status=$?
if [[ $status -ne 0 ]]; then
  echo "Error submitting job $jobid for $cmd" 2>&1
  exit 1
fi
echo "Submitted job $jobid for $cmd" 2>&1
echo $jobid
