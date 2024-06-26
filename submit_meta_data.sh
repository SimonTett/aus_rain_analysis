#!/usr/bin/env bash
# submit scripts to extract metadata from ppi files
time_str=$(date +"%Y%m%d_%H%M%S")
history_file="history_$(basename ${0}).txt"
echo "history file is: ${history_file}"
echo "${time_str}: $0 $*" >> "${history_file}"
# Loop through all the positional parameters
while (( "$#" )); do
  case "$1" in
    --years)
      shift
      extra_args+=' --years '
      while (( "$#" )) && [[ $1 != -* ]]; do
        pp_extra_args+="$1 "
        shift
      done
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
        extra_args+="--log_file ${2}"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --outdir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        extra_args+=" --outdir  $2"
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
walltime='2:00:00'
project=wq02
memory=10GB
ncpus=1 # single core job as largely doing IO
cmd="./process_extract_metadata.py ${site} ${extra_args}"
name="${site}_ext_meta"
log_dir="/scratch/wq02/st7295/radar/log"
gen_script () {
    # function to generate PBS script

    log_file="${log_dir}/${name}_${time_str}"
    mkdir -p ${log_dir}
    # print out the PBS commands
    cat <<EOF
#PBS -P ${project}
#PBS -q normalbw
#PBS -l walltime=${walltime}
#PBS -l storage=gdata/rq0+gdata/hh5+gdata/ua8
#PBS -l mem=${memory}
#PBS -l ncpus=${ncpus}
#PBS -l jobfs=10GB
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
    echo "post-processing script is:"
    gen_script
else
  job_name=$(gen_script | qsub ${qsub_args} -) # generate and submit script
  echo "Submitted post-processing job with name ${name} using qsub ${qsub_args}" >&2
  echo $job_name
fi



