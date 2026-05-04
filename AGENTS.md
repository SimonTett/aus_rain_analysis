# AGENTS.md

## Scope and Source of Truth
- This repo is a radar-rainfall processing pipeline plus analysis/plot scripts; primary logic lives in `ausLib.py` and `process_submit/`.
- Existing AI-specific instruction files were not found; conventions here are inferred from `README.md`, `process_submit/README.md`, scripts, and config JSON.
- Treat `ausLib.py` as the shared API for CLI behavior, logging, job submission, and I/O helpers.

## Big Picture Data Flow
- Pipeline order is: `process_reflectivity.py` -> `process_seas_avg_mask.py` -> `process_events.py` -> `process_gev_fits.py`.
- `process_submit/submit_process_reflectivity.sh` launches per-year reflectivity jobs; `process_submit/submit_post_process.sh` chains downstream jobs with `--holdafter` dependencies.
- Reflectivity stage reads zipped NetCDF volumes (`hist_gndrefl` or `rainfields3`), optional coarsening/conversion to rain, and writes monthly summaries.
- Seasonal stage aggregates to DJF (or monthly), applies quality masks (beam blockage, elevation/land, sample sufficiency, threshold artifacts), and writes `seas_mean_*.nc`.
- Events stage groups maxima by day-like bins (reference `1970-01-01T14:00`), adds covariates/topography, outputs `events_*.nc`.
- GEV stage calls `R_python.gev_r.xarray_gev` (R dependency) and writes fit NetCDF plus text summaries.

## Critical Workflows
- HPC-first workflow assumes Gadi modules and env vars from `setup.sh` (`AUSRAIN`, `PATH`, `PYTHONPATH`, R libs, dask optimiser).
- Typical full submit path: `process_submit/submit_ref_plus_pp.sh <Site> [options]`.
- Reflectivity-only submit path: `process_submit/submit_process_reflectivity.sh <Site> --years <start> <end> ...`.
- Post-process only path (existing summaries): `process_submit/submit_post_process.sh <summary_dir> [--region ...] [--covariates ...]`.
- JSON defaults in `config_files/*.json` are merged by `ausLib.process_std_arguments`; CLI flags override JSON/defaults.
- `process_events.json` uses queue `copyq` because ACORN retrieval in `ausLib.read_acorn` requires internet access.

## Project Conventions to Follow
- New processing CLIs should call `ausLib.add_std_arguments(parser, ...)` and then `ausLib.process_std_arguments(args, ...)`.
- Standard flags used everywhere: `--overwrite`, `-v/--verbose`, `--log_file`, optional `--submit` batch options.
- Every output dataset is stamped with `program_name`, UTC timestamp, and `program_args` (see all `process_submit/process_*.py`).
- Use `ausLib.write_out(...)` for NetCDF output so time encoding/compression/attrs remain consistent.
- Site naming/IDs and machine-specific paths are centralized in `ausLib.py` (`site_numbers`, hostname switch for `data_dir`, etc.).
- Be careful with dimensions: many routines assume `x`,`y`,`time`,`resample_prd`,`EventTime`,`quantv` names exactly.

## Integrations and Gotchas
- Host-dependent path logic in `ausLib.py` raises `NotImplementedError` on unknown machines; local runs may need hostname/path support added.
- Batch backend is auto-detected (`sbatch` or `qsub`) in `ausLib.detect_batch_system`; submission script text is generated in Python.
- Dask is optional but often discouraged in README/scripts for I/O-bound steps; `process_gev_fits.py` explicitly blocks dask+R.
- External dependencies include BoM ACORN HTTP (`requests`), ERA5 files under `ausLib.data_dir`, and CBB/DEM ancillary files per site.
- `ausLib.read_radar_zipfile` currently contains a `breakpoint()`; unattended pipeline runs can halt if that codepath executes.
- Cross-repo dependency: GEV fitting expects importable `R_python` package (see sibling `commonLib_gh/R_python/README.md`).

## High-Value Files to Read Before Editing
- `ausLib.py`
- `process_submit/process_reflectivity.py`
- `process_submit/process_seas_avg_mask.py`
- `process_submit/process_events.py`
- `process_submit/process_gev_fits.py`
- `process_submit/submit_process_reflectivity.sh` and `process_submit/submit_post_process.sh`

