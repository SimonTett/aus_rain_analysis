#!/usr/bin/env bash
# submit scripts to process reflectivity data
site=$1 ; shift # get site name
if [[ $# -eq 0 ]]
then
    outdir="/scratch/wq02/st7295/summary_reflectivity"
else
    outdir=$1 ; shift # get output directory
fi

years="1995 2000 2005 2010 2015 2020" # want those years.
walltime='08:00:00'
project=wq02
memory=15GB
ncpus=4 # WIll have a few CPs coz of memory and so will use then
time_str=$(date +"%Y%m%d_%H%M%S")
gen_script () {
    # function to generate PBS script
    site=$1; shift
    year=$1 ; shift
    outdir=$1 ; shift
    log_dir=$1 ; shift
    mkdir -p ${log_dir}
    end_year=$((year+5))
    cmd="./process_reflectivity.py ${site} ${outdir}  -v --year ${year} ${end_year} --dask --no_over_write --resample  ${resample} "
    log_file="${log_dir}/proc_refl_${site}_{$year}_${time_str}"
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
resample='30min 1h 2h 4h 8h'
for year in ${years}
  do
  log_dir="/scratch/${project}/st7295/radar_log/${site}"
  gen_script "${site}" "${year}" "${outdir}" "${log_dir}" | qsub - # generate and submit script
done
