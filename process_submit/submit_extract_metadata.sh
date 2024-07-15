#!/usr/bin/env bash
# submit scripts to extract metadata from ppi files
time_str=$(date +"%Y%m%d_%H%M%S")
base_dir="/scratch/wq02/st7295/radar/site_data/"
site=$1 ; shift # get site name
if [[ -z ${site} ]]
then
    echo "Error: Site name is required" >&2
    exit 1
fi
outdir="${base_dir}/${site}_metadata"
log_file="${outdir}/run_logs/extract_metadata_${site}_${time_str}.log"
pbs_base="${outdir}/pbs_logs/extract_metadata_${site}_${time_str}"
job_name="extmd_${site}"
submit_opts=" --submit --json_submit $AUSRAIN/config_files/process_extract_metadata.json --log_base $pbs_base"
submit_opts+=" --log_file ${log_file} --job_name ${job_name} "
cmd="process_extract_metadata.py  ${site} --outdir ${outdir} $*  ${submit_opts}"
jobid_event=$($cmd)
status=$?
if [[ $status -ne 0 ]]; then
  echo "Error submitting job $jobid_event for $cmd" 2>&1
  exit 1
fi
echo "Submitted job $jobid_event for $cmd" 2>&1




