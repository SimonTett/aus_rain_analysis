#!/usr/bin/env bash
# submit scripts to post-process the radar data

base_dir="/scratch/wq02/st7295/radar"
# Loop through all the positional parameters
while (( "$#" )); do
  case "$1" in
    --input_dir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        extra_args+=" --input_dir ${2}"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --name)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
         extra_args+=" --name ${2}"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --outdir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        extra_args+=" --outdir ${2}"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --cbb_dir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        extra_args+=" --cbb_dir ${2}"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --years)
      shift
      extra_args+=' --years '
      while (( "$#" )) && [[ $1 != -* ]]; do
        extra_args+="$1 "
        shift
      done
      ;;
    --season)
      shift
      extra_args+=" --season $1"
      seas_str=$1
      shift
      ;;
    --dask)
      extra_args+=" --dask"
      shift
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
    --verbose)
      extra_args+=" --verbose"
      shift
      ;;
    --dryrun)
      dryrun='True'
      echo "Dry run only" >&2
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

if [[ -z ${name} ]]
then
    name=${site}
fi

memory=15GB
project=wq02
walltime='01:00:00'
ncpus=1
job_name="pp_${name}_${time_str}"
log_dir="${base_dir}/log/"
mkdir -p ${log_dir}
log_file=${log_dir}/"pp_${name}_${time_str}"
echo "post-processing log file is: ${log_file}" >> ${history_file}
cmd="./run_post_process.sh ${site} ${extra_args}" # just pass args through
gen_script () {
  # function to generate PBS script for post-processing. This needs access to the internet to download
  #  acorn data.  So runs on copyq. If the data is already downloaded then it can run on normalbw.
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
#PBS -N ${job_name}
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
    gen_script >&2
else
  job_name=$(gen_script | qsub ${qsub_args} -) # generate and submit script
  echo "Submitted post-processing job with name ${job_name} submitted with qsub args ${qsub_args}" >&2
fi