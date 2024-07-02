#!/usr/bin/env bash
# submit  all post-processing, All args are passed straight through to the individual submit_post_process.sh scripts

args=$*
dirs=$(ls -1 /scratch//wq02/st7295/radar/summary/ | grep rain | grep -v coord)
for dir in $dirs; do
  site=$(echo $dir | cut -d'_' -f1)
  echo "submitting post process for $dir"
  ./submit_post_process.sh ${site} --name $(basename $dir) --outdir /scratch/wq02/st7295/radar/processed/${dir} ${args}
done
