#!/usr/bin/env bash
# This script is used to generate events data from the radar monthly processed files.
conda activate /home/z3542688/analysis/aus_rain_analysis/mamba_rain_env
export PYTHONPATH=~/analysis/common_lib:.:$PYTHONPATH

site=$1 ; shift
root_dir=$1 ; shift # where files will be put.
files=$* # get all the files passed as arguments
CBB_dir="${root_dir}/${site}/cbb_dir"
mn_file="${root_dir}/${site}/seas_mean_${site}_DJF.nc"
event_file="${root_dir}/${site}/events_${site}_DJF.nc"
fit_root="${root_dir}/${site}/gev_fits/${site}"
extra_args='-v' # include dask is want multiprocessing
./process_beam_blockage.py "${site}" --output_dir "$CBB_dir"  ${extra_args}  # generate the DEM and beam blockage data for the site.
CBB_files="${CBB_dir}/*.nc"
# generate the seasonal average mask for the site.
./process_seas_avg_mask.py ${files} "${mn_file}"  --cbb_dem_files ${CBB_files} --site $site ${extra_args}
./process_events.py ${mn_file} ${event_file} --cbb_dem_files ${CBB_files} ${extra_args} # generate the events data for the site.
./process_gev_fits.py ${event_file}  ${fit_root} ${extra_args} # generate the GEV fits for the site.