#!/usr/bin/env bash
# Post-process the radar data. First argument is the directory of the radar data to process
# Some other optional args are --region x0 y0 x1 y1 reg_name -- region to extract to and name of region.
# If this is provided then code does nto subit job to do seasonal meaning/masking as assumes this has already been done.
#  The output events and gev fits are created in a new directory based on the regional name.
# --holdafter jobs -- run first post-processing script after the job(s) have completed.
# Need to have run setup.sh before running this script.
if [[ -z $AUSRAIN ]]; then
  echo "Error: Must define AUSRAIN first. Have you done . setup.sh ??" >&2
  exit 1
fi
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
        echo "Using region assuming seasonal mean file already exists." >&2
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
if [[ -n "${region_name}" ]]; then
  if [[ ! -f ${sm_file} ]]; then
    echo "Error: seasonal mean file ${sm_file} does not exist" >&2
    echo "Run process_seas_avg_mask.py first to create this file" >&2
    exit 1
  fi
  name+=$region_name
  process_dir+=$region_name
  echo "Using region $region_name to modify process_dir and name" >&2
fi
event_file="${process_dir}/events_seas_mean_${name}_${seas_str}.nc"
gev_dir="${process_dir}/fits"
pbs_log_dir="${process_dir}/pbs_logs"
run_log_dir="${process_dir}/run_logs"
time_str=$(date +"%Y%m%d_%H%M%S")
# run the meaning
jobid_mean=""
if [[ -z "${region_name}" ]]; then
  job_name="smn_${name}"
  log_file=process_seas_avg_mask_${name}_${time_str}
  submit_opts=" --submit"
  submit_opts+=" --json_submit ${AUSRAIN}/config_files/process_seas_avg_mask.json --log_base ${pbs_log_dir}/${log_file}"
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
  hold_after="--holdafter ${jobid_mean}" # hold after only if run the mean processing
fi
# event processing
job_name="ev_${name}"
log_file=process_events_${name}_${time_str}
submit_opts=" --submit --json_submit  ${AUSRAIN}/config_files/process_events.json --log_base ${pbs_log_dir}/${log_file}"
submit_opts+=" --log_file ${run_log_dir}/${log_file}.log --job_name ${job_name} "
submit_opts+=" ${hold_after} "
cmd="process_events.py  ${sm_file} ${event_file}  ${extra_args} ${region_args} ${submit_opts}"
jobid_event=$($cmd)
hold_after=" --holdafter ${jobid_event}" # hold after evbent processing
status=$?
if [[ $status -ne 0 ]]; then
  echo "Error submitting job $jobid_event for $cmd" 2>&1
  exit 1
fi
echo "Submitted job $jobid_event for $cmd" 2>&1
# GEV processing
job_name="gev_${name}"
log_file=process_gev_fits_${name}_${time_str}
submit_opts=" --submit --json_submit  ${AUSRAIN}/config_files/process_gev_fits.json --log_base ${pbs_log_dir}/${log_file}"
submit_opts+=" --log_file ${run_log_dir}/${log_file}.log --job_name ${job_name} "
submit_opts+=" ${hold_after} "
cmd="process_gev_fits.py ${event_file} --outdir ${gev_dir} --nsamples=100 --bootstrap=100 ${extra_args} ${submit_opts} "
jobid_gev=$($cmd)
status=$?
if [[ $status -ne 0 ]]; then
  echo "Error submitting job $jobid_gev for $cmd" 2>&1
  exit 1
fi
echo "Submitted job $jobid_gev for $cmd" 2>&1
echo "${jobid_mean}" "${jobid_event}" "${jobid_gev}" # return list of all jobs submitted.

