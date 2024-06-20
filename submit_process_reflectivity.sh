#!/usr/bin/env bash
# submit scripts to process reflectivity data and then (once all ran) submit post-processing
extra_args="" # extra args
name=''
site=''
pp_root_dir='/scratch/wq02/st7295/radar/processed'
max_root_dir="/scratch/wq02/st7295/radar/summary"

max_extra_args=""

# Loop through all the positional parameters
while (( "$#" )); do
  case "$1" in
    --threshold)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        max_extra_args+=" --threshold $2"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --min_fract_avg)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        max_extra_args+=" --min_fract_avg $2"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --to_rain)
      if [ -n "$3" ] && [ ${3:0:1} != "-" ]; then
        max_extra_args+=" --to_rain $2 $3"
        shift 3
      else
        echo "Error: Arguments for $1 are missing" >&2
        exit 1
      fi
      ;;
    --extract_coords_csv)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        max_extra_args+=" --extract_coords_csv $2"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --name)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        name=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --pp_root_dir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        pp_root_dir=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --ccb_dir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        pp_extra_args+="--cbb_dir $2"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --max_root_dir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        max_root_dir=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --years)
      shift
      pp_extra_args+=' --years '
      while (( "$#" )) && [[ $1 != -* ]]; do
        pp_extra_args+="$1 "
        shift
      done
      ;;
    --season)
      shift
      pp_extra_args+=" --season $1"
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
      dryrun=True
      shift
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
if [[ -n ${max_root_dir} ]]
then
    max_root_dir="/scratch/wq02/st7295/radar/summary/${name}"
fi
if [[ -n ${pp_root_dir} ]]
then
    pp_root_dir="/scratch/wq02/st7295/radar/processed/${name}"
fi
years_to_gen="1995 2000 2005 2010 2015 2020" # want those years.
walltime='12:00:00'
project=wq02
memory=25GB
ncpus=4 # WIll have a few CPs coz of memory and so will use then
time_str=$(date +"%Y%m%d_%H%M%S")
resample='30min 1h 2h 4h 8h'
region="-125 125 125 -125" # region to extract
coarse=4
max_extra_args+=' --dask' # use dask for processing
if [[ -n "$resample" ]]
then
    max_extra_args+=" --resample ${resample}"
fi
if [[ -n "$region" ]]
then
    max_extra_args+=" --region ${region}"
fi
if [[ -n "$coarse" ]]
then
    max_extra_args+=" --coarsen ${coarse} ${coarse}"
fi

gen_max_script () {
    # function to generate PBS script for processing the max.

    site=$1; shift
    name=$1; shift
    year=$1 ; shift
    max_root_dir=$1 ; shift
    log_dir=$1 ; shift
    mkdir -p ${log_dir}
    years=$(seq -s ' ' ${year} $((year+4))) # five years at a time
    cmd_log_file="${max_root_dir}/log/${name}_${year}_${time_str}"
    cmd="./process_reflectivity.py ${site} ${max_root_dir}  --years ${years}  --log_file ${cmd_log_file} ${max_extra_args} ${extra_args}"
    log_file="${log_dir}/proc_refl_${name}_${year}_${time_str}"
    job_name=${name:0:6}_${year}
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

job_depend=''
for year in ${years_to_gen}
  do
  log_dir="${max_root_dir}/${name}/${name}_pbs_log"
  script_cmd="gen_max_script ${site} ${name} ${year} ${max_root_dir} ${log_dir}"
  echo "Running ${script_cmd}"
  if [[ -n "$dryrun" ]]
  then
      echo "Would submit job for ${site} ${name} and year ${year} with log dir ${log_dir}. Script is:"
      eval $script_cmd
  else
    echo "Submitting max job"
    job_name=$(gen_max_script ${site} ${name} ${year} ${max_root_dir} ${log_dir} | qsub - ) # generate and submit script
    echo "Submitted job with name ${job_name} "
    job_depend+=":${job_name}"
  fi

done

# now submit the post-processing with a holdafter for the processing jobs.
memory=15GB
job_name="pp_${name}_${time_str}"
log_file="${pp_root_dir}/log/pp_${name}_${time_str}"
cmd="./run_post_process.sh --input_dir ${max_root_dir} --name ${name} --site ${site} --root_dir ${pp_root_dir}  ${pp_extra_args} ${extra_args}"
gen_pp_script () {
cat <<EOF
#PBS -P ${project}
#PBS -q normalbw
#PBS -l walltime=${walltime}
#PBS -l storage=gdata/rq0+gdata/hh5+gdata/ua8
#PBS -l mem=${memory}
#PBS -l ncpus=1
#PBS -l jobfs=10GB
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
    gen_pp_script
else
  job_name=$(gen_pp_script | qsub -W "depend=afterok${job_depend}" -) # generate and submit script
  echo "Submitted post-processing job with name ${job_name} dependant on ${job_depend}"
fi
