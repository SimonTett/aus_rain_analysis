# Scripts to process and submit data.

## Processing python scripts
All should have common options:

    --overwrite (overwrite existing data)

    -v (verbose: once = INFO, twice= DEBUG)

    --log_file -- have a log file for the logging output. (Need to have some verbosity to get anything )

   and a bunch to do with job submission.

Some will have an option --dask to use dask.  However, most of the code appears to be sufficiently I/O bound that dask doesn't help.
The python scripts are:

* process_beam_blockage.py -- processes the beam blockage data. Only needs to run once per site.
* process_reflectivity -- processes the reflectivity data. Produces monthly max, mean and a bunch of other information.
* process_seas_avg_mask.py -- processes the monthly  data produced by process_reflectivity. AGenerates seasonal values with some QC
* process_events.py -- processes the seasonal data to events.
* process_gev_fits.py -- processes the events data to give GEV fits.


## Submission scripts
There are all bash scripts and generally submit a bunch of jobs to the queue and return (on stdout) the job ids.
They call the processing scripts.
Generally any command line arguments given are passed through to the processing scripts they call,

1) submit_beam_blockage.sh -- submits the beam blockage processing.
2) submit_reflectivity.sh -- submits the reflectivity processing. A job per year from 1997-2022 is submitted.
3) submit_post_process.sh  -- submits the post-processing scripts to  run in the right order.
     Calls process_seas_avg_mask.py, process_events.py and process_gev_fits.py
