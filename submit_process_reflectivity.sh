#!/usr/bin/env bash
# submit scripts to process reflectivity data
site=$1 ; shift # get site name
years="1995 2000 2005 2010 2015 2020" # want those years.
walltime='08:00:00'
project=wq02
memory=50GB
ncpus=2 # I/O so not much point in being parallel. Memory will drive it.

gen_script () {
    # function to generate PBS script
    name=$1; shift
    year=$1 ; shift
    log_dir=$1 ; shift
    end_year=$((year+5))
    cmd="./process_reflectivity.py $site -v -v --year $year $end_year --resample $resample --no_over_write"
    log_file="${log_dir}/${name}/processed_${name}.log"
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
#PBS -N pref_${name}_${year}
#PBS -o /home/561/st7295/aus_rain_analysis/pbs_output/${name}_${year}.out
#PBS -e /home/561/st7295/aus_rain_analysis/pbs_output/${name}_${year}.err
export TMPDIR=\$PBS_JOBFS
cd /home/561/st7295/aus_rain_analysis || exit # make sure we are in the right directory
. ./setup.sh # setup software and then run the processing
echo ${cmd}
result=\$($cmd)
echo \$result
EOF
    return 0
    }
resample='30min 1h 2h 4h 8h'
for year in ${years}
  do

  log_dir="/scratch/${project}/st7295/radar_log"
  gen_script ${site} ${year} ${log_dir} | qsub - # generate and submit script
done
