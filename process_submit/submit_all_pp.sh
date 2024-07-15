#!/usr/bin/env bash
# submit  all post-processing, All args are passed straight through to the individual submit_post_process.sh scripts
# should have setup.sh run first to get the environment set up
args=$*
dirs=$(ls -1d /scratch/wq02/st7295/radar/summary/* | grep rain | grep -v coord) # directories we want to process.
for dir in $dirs; do
  echo "submitting post process for $dir"
  submit_post_process.sh $dir $args
done
