#!/usr/bin/env bash
# submit scripts to compute beam blockage factors. Needs to run in a q with access to the internet.
time_str=$(date +"%Y%m%d_%H%M%S")
history_file="history_$(basename ${0}).txt"
echo "history file is: ${history_file}" >&2
echo "${time_str}: $0 $*" >> "${history_file}"

# Loop through all the positional parameters
while (( "$#" )); do
  case "$1" in
  --outdir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        extra_args+=" --outdir ${2}"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
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
    --log_file)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        extra_args+=" --log_file ${2}"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --dryrun)
      dryrun=True
      shift
      ;;
    --holdafter)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        qsub_args+=" -W depend=afterany:${2}"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --purpose)
      shift
      purpose=''
      while (( "$#" )) && [[ $1 != -* ]]; do # eat all the arg. Purpose doesn't do anything. Needed for history.
        purpose+="$1 "
        shift
      done
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
site=$1 ; shift # get site name
if [[ -z ${site} ]]
then
    echo "Error: Site name is required" >&2
    exit 1
fi
# stuff for PBS script
walltime='1:00:00'
project=wq02
memory=20GB # actually need 20GB
ncpus=1 # single core job as largely doing IO
cmd="./process_beam_blockage.py ${site} ${extra_args}"
name="${site}_beam"
log_dir="/scratch/wq02/st7295/radar/log"
mkdir -p ${log_dir} # make the log directory if it doesn't exist
log_file="${log_dir}/${name}_${time_str}"
gen_script () {
    # function to generate PBS script
  cat <<EOF
#PBS -P ${project}
#PBS -q copyq
#PBS -l walltime=${walltime}
#PBS -l storage=gdata/rq0+gdata/hh5+gdata/ua8
#PBS -l mem=${memory}
#PBS -l ncpus=${ncpus}
#PBS -l jobfs=20GB
#PBS -l wd
#PBS -m abe
#PBS -M simon.tett@ed.ac.uk
#PBS -N ${name}
#PBS -o ${log_file}.out
#PBS -e ${log_file}.err
export TMPDIR=\$PBS_JOBFS
cd /home/561/st7295/aus_rain_analysis || exit # make sure we are in the right directory
. ./setup.sh # setup software and then run the processing
echo Cmd is ${cmd}
result=\$($cmd)
echo \$result
EOF
return 0
}

if [[ -n "$dryrun" ]]
then
    echo "beam blockage script is:"
    gen_script
    echo -e "++++++++++++++++++ \n qsub args are ${qsub_args}"
else
  job_name=$(gen_script | qsub ${qsub_args} -) # generate and submit script
    stat=$?
  if [[ $stat -ne 0 ]]
  then
      echo "Error: qsub returned non-zero status ${stat}" >&2
      exit $stat
  fi
  echo $job_name
fi



