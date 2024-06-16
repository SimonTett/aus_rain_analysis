#!/usr/bin/env bash
# submit scripts to process reflectivity data
site=$1 ; shift # get site name
if [[ $# -eq 0 ]]
then
    outdir="/scratch/wq02/st7295/summary_reflectivity"
else
    outdir=$1 ; shift # get output directory
fi

years_to_gen="1995 2000 2005 2010 2015 2020" # want those years.
walltime='12:00:00'
project=wq02
memory=25GB
ncpus=4 # WIll have a few CPs coz of memory and so will use then
time_str=$(date +"%Y%m%d_%H%M%S")
extra_args="--to_rain 0.02567 0.6875 --extract_coords_csv meta_data/${site}_close.csv"
resample='30min 1h 2h 4h 8h'
region="-100 100 100 -100" # region to extract
if [[ -n "$resample" ]]
then
    extra_args="${extra_args} --resample ${resample}"
fi
if [[ -n "$region" ]]
then
    extra_args="${extra_args} --region ${region}"
fi
extra_args="${extra_args} --min_fract_avg 0.75 --threshold 10."
gen_script () {
    # function to generate PBS script

    site=$1; shift
    year=$1 ; shift
    outdir=$1 ; shift
    log_dir=$1 ; shift
    mkdir -p ${log_dir}
    years=$(seq -s ' ' ${year} $((year+4))) # five years at a time
    cmd_log_file="${outdir}/log/${site}_${year}_${time_str}"
    cmd="./process_reflectivity.py ${site} ${outdir}  -v -v --years ${years} --dask --no_over_write  --coarsen 4 4  --log_file ${cmd_log_file} --min_fract_avg 0.75 --dbz_range 15 55"
    if [[ -n "$extra_args" ]]
    then
        cmd="${cmd} ${extra_args}"
    fi
    log_file="${log_dir}/proc_refl_${site}_${year}_${time_str}"
    job_name=${site:0:6}_${year}
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

for year in ${years_to_gen}
  do
  log_dir="${outdir}/${site}_pbs_log"
  gen_script "${site}" "${year}" "${outdir}" "${log_dir}" | qsub - # generate and submit script
done
