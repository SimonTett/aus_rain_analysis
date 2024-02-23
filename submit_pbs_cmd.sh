#!/usr/bin/env bash
# run a shell scrip to submit a processing job (or several)
# sett walltime & ncpus to change time/core resources
# Melbourne takes  about 4 minutes  to process a year.
name=$1; shift
walltime='08:00:00'
project=wq02
ncpus=7 # number of cores to use. sweet spot seems to be 7! (1/4 of a node)
# Melbourne 2016 1 year
# cores -- wallclock cpu (all in secs) par eff Units Mem (Gbytes)
# 28 162 1262 0.28
# 16 176 1071 0.38  1.01 7.9
# 14 176 1045 0.42  0.86 7.6
#  8 207  930 0.56  0.57 6.3
#  7 223  931 0.59  0.54 6.3
#  4 290  831 0.71  0.67 5.9

if [[ $# -gt 0 ]]
then
    year=$1 ; shift
    walltime='00:15:00' # one year takes about 5 minutes.
fi

declare -A radar_numbers # translation from names to numbers for the radars
radar_numbers=([adelaide]=46 [melbourne]=2 [wtakone]=52 [sydney]=3 [brisbane]=50 [canberra]=40 \
			 [cairns]=19 [mornington]=36 [grafton]=28 [newcastle]=4 [gladstone]=23)

gen_script () {
    # function to generate PBS script
    name=$1; shift
    if [[ $# -gt 0 ]] ;    then
	    year=$1 ; shift
    fi
    number=${radar_numbers[${name}]}
    if [[ -z "${number}" ]] ; then
	    echo "failed to find ${name} in radar_numbers table. Exiting"
    	exit 1
    fi
    output_dir="/scratch/${project}/st7295/radar"

    set -o noglob # turn of globbing here
    pattern="${number}_*rainrate.nc"
    output="${output_dir}/${name}/processed_${name}.nc"
    log_file="${output_dir}/${name}/processed_${name}.log"
    extra="--year_chunk --cache"
    if [[ -n "${year}" ]] ;  then
	    pattern="${number}_${year}*rainrate.nc"
	    output="${output_dir}/${name}/processed_${year}_${name}.nc"
	    log_file="${output_dir}/${name}/processed_${year}_${name}.log"
	    extra=""
    fi
    # print out the PBS commands
    cat <<EOF
#PBS -P ${project}
#PBS -q normalbw
#PBS -l walltime=${walltime}
#PBS -l storage=gdata/rq0+gdata/hh5+gdata/ua8
#PBS -l mem=60GB
#PBS -l ncpus=${ncpus}
#PBS -l jobfs=2GB
#PBS -l wd
#PBS -m abe
#PBS -M simon.tett@ed.ac.uk
#PBS -N pradar_${name}
#PBS -o /home/561/st7295/aus_rain_analysis/pbs_output/${name}.out
#PBS -e /home/561/st7295/aus_rain_analysis/pbs_output/${name}.err
cd /home/561/st7295/aus_rain_analysis || exit # make sure we are in the right directory
. ./setup.sh # setup software and then run the processing
set -o noglob # noglob -- leave python to expand
EOF
    cmd="'./comp_radar_max.py /g/data/rq0/level_2/${number}/RAINRATE/${pattern} ${output}  --verbose --dask --log_file ${log_file} ${extra}'"
    echo cmd=${cmd}
    cat <<'EOF'
echo ${cmd}
result=$($cmd)
echo $result
EOF
    return 0
    }

echo "Name is $name"

if [[ -n "${year}" ]] ; then
    year_to_use=""
else
    year_to_use=${year}
fi

if [[ ${name} = 'all' ]]; then
    echo  "running all cases"
    for my_name in "${!radar_numbers[@]}" ;     do
      echo "====== $my_name $year_to_use ==============="
      gen_script $my_name $year_to_use  | qsub - # generate and submit script
      echo "========================"
      sleep 10
    done
else
    echo "====== $name $year_to_use ==============="
    gen_script $name $year_to_use | qsub - # generate and submit script
    echo ==============
fi
