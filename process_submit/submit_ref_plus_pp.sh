#!/usr/bin/env bash
# submit process reflectivity jobs AND post-processing jobs.
# post-processing is submitted held after the reflectivity processing is complete.
# all arguments are passed to submit_process_reflectivity.sh
# example usage:
# ./submit_ref_ann_pp.sh Melbourne --calibration melbourne --years 1997 2022 ---holdafter 1234
# Check have AUSRAIN set
if [[ -z "$AUSRAIN" ]]
then
    echo "AUSRAIN environment variable not set. Please set it to the aus_rain_analysis directory."
    exit 1
fi
# extract common args
common_args=""
while (( "$#" )); do
  case "$1" in
     --dryrun|-v|--overwrite)
        common_args+=" $1"
        shift
        ;;
     *)
      args+=" $1"
      shift
      ;;
  esac
done
cmd="submit_process_reflectivity.sh  ${args} --return_dir ${common_args}"
result=$($cmd)
status=$?
if [[ $status -ne 0 ]]; then
  echo "Error: ${cmd} failed with status $status" >&2
  exit $status
fi
for word in $result; do
  if [[ -z ${outdir} ]]; then
    outdir=${word}
  else
    jobids+="${word} "
  fi
done
echo "Submitted reflectivity processing jobs with jobids: $jobids" >&2
# submit post-processing
cmd="submit_post_process.sh ${outdir} --holdafter ${jobids} ${common_args}"
pp_jobids=$($cmd)
status=$?
if [[ $status -ne 0 ]]; then
  echo "Error: ${cmd} failed with status $status" >&2
  exit $status
fi
echo "Submitted post-processing jobs with pp_jobids: $pp_jobids" >&2
echo "${jobids} ${pp_jobids}" # return list of all jobs submitted.



