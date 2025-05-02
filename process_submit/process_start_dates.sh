#!/usr/bin/env bash

# Time ranges for the following
#Wtakone -- 2014-12-01 (When calibration from GPM is available)
#Gladstone -- 2005-12-01 (Before then is a WF44 radar which we
#remove. Data for 2004-12-01 a bit dodgy as well)
#Melbourne -- 2003-12-01 (Looks dodgy when compared to mean rainfall)

## scripts (and arguments) to run this processing.
radius="125e3"
covariates="temperature dewpoint sample_resolution fraction"
submit_post_process.sh \
    /scratch/stett2/radar/processed/Gladstone_rain_brisbane \
    --time_range 2005-12-01 -v -v --covariates $covariates --radius $radius


submit_post_process.sh \
    /scratch/stett2/radar/processed/Gladstone_rain_melbourne \
    --time_range 2005-12-01 -v -v --covariates $covariates --radius $radius

submit_post_process.sh \
    /scratch/stett2/radar/processed/Melbourne_rain_brisbane \
    --time_range 2003-12-01 -v -v --covariates $covariates --radius $radius

submit_post_process.sh \
    /scratch/stett2/radar/processed/Melbourne_rain_melbourne \
    --time_range 2003-12-01 -v -v --covariates $covariates --radius $radius

submit_post_process.sh \
    /scratch/stett2/radar/processed/Wtakone_rain_brisbane  \
    --time_range 2014-12-01 -v -v --covariates $covariates --radius $radius

submit_post_process.sh \
    /scratch/stett2/radar/processed/Wtakone_rain_melbourne \
    --time_range 2014-12-01 -v -v --covariates $covariates --radius $radius

