#!/usr/bin/env python
# created by chatgpt co-pilot by converting submit_post_process.sh to python
import argparse
import os
import errno
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Constants
EEXIST: int = errno.EEXIST # changed
AUSRAIN: Optional[str] = os.getenv("AUSRAIN")
if not AUSRAIN:
    raise EnvironmentError("Error: AUSRAIN environment variable is not defined. Run '. setup.sh' first.")

def submit_job(
    cmd: List[str],
    log_file: str,
    job_name: str,
    hold_after: Optional[list[str]],
    json_submit: Path,
    log_base: Path,
    log_dir: Path
) -> str:
    """
    Submits a job using the provided command and options.

    Args:
        cmd (List[str]): The base command to execute.
        log_file (str): The name of the log file.
        job_name (str): The name of the job.
        hold_after (Optional[str]): Dependencies for the job to hold after.
        json_submit (Path): Path to the JSON submission configuration.
        log_base (Path): Base directory for logs.
        log_dir (Path): Directory for the log file.

    Returns:
        str: The jobid of  the submitted job.

    Raises:
        RuntimeError: If the job submission fails.
    """
    submit_opts: List[str] = [ # options for submitting the job
        "--submit",
        '--json_submit',json_submit,
        "--log_base",log_base,
        "--log_file",str(log_dir / f'{log_file}.log'),
        "--job_name",job_name,
    ]
    if hold_after:
        submit_opts.append('--holdafter')
        submit_opts.extend(hold_after)

    full_cmd: List[str] = cmd + submit_opts
    try:
        result = subprocess.run(full_cmd, check=True, capture_output=True, text=True)
        job_id: str = result.stdout.strip()
        print(f"Submitted job {job_id} for {' '.join(cmd)}")
        return job_id
    except subprocess.CalledProcessError as e:
        if e.returncode == EEXIST:
            print("Output file already exists. Skipping job submission.")
            return ""
        else:
            raise RuntimeError(f"Error submitting job for {' '.join(cmd)}: {e.stderr}")

def process_mean_mask(
    summary_dir: Path,
    sm_file: Path,
    nomask_file: Path,
    args: List[str],
    extra_args: List[str],
    hold_after: Optional[List[str]],
    log_file: str,
    job_name: str,
    json_submit: Path,
    log_base: Path,
    log_dir: Path
) -> str:
    """
    Processes seasonal mean and mask data.

    Args:
        summary_dir (Path): Directory containing radar data.
        sm_file (Path): Path to the seasonal mean file.
        nomask_file (Path): Path to the no-mask file.
        args (List[str]): Additional arguments for seasonal masking.
        extra_args (List[str]): Extra arguments for processing.
        hold_after (Optional[List[str]]): Dependencies for the job to hold after.
        log_file (str): Log file name.
        job_name (str): Job name.
        json_submit (Path): Path to the JSON submission configuration.
        log_base (Path): Base directory for logs.
        log_dir (Path): Directory for the log file.

    Returns:
        str: The submitted job ID.
    """
    cmd: List[str] = [
        "process_seas_avg_mask.py",
        str(summary_dir),
        "--output", str(sm_file),
        "--no_mask_file", str(nomask_file)
    ] + args + extra_args
    return submit_job(cmd, log_file, job_name, hold_after, json_submit, log_base, log_dir)

def process_events(
    seas_mean_file: Path,
    event_file: Path,
    extra_args: List[str],
    region_args: List[str],
    hold_after: Optional[str],
    log_file: str,
    job_name: str,
    json_submit: Path,
    log_base: Path,
    log_dir: Path
) -> str:
    """
    Processes events from seasonal mean data.

    Args:
        seas_mean_file (Path): Path to the seasonal mean file.
        event_file (Path): Path to the events file.
        extra_args (List[str]): Extra arguments for processing.
        region_args (List[str]): Region-specific arguments.
        hold_after (Optional[str]): Dependencies for the job to hold after.
        log_file (str): Log file name.
        job_name (str): Job name.
        json_submit (Path): Path to the JSON submission configuration.
        log_base (Path): Base directory for logs.
        log_dir (Path): Directory for the log file.

    Returns:
        str: The holdafter string for the submitted job.
    """
    cmd: List[str] = [
        "process_events.py",
        str(seas_mean_file),
        str(event_file)
    ] + extra_args + region_args
    return submit_job(cmd, log_file, job_name, hold_after, json_submit, log_base, log_dir)

def process_gev(
    event_file: Path,
    gev_dir: Path,
    gev_args: List[str],
    extra_args: List[str],
    hold_after: Optional[list[str]],
    log_file: str,
    job_name: str,
    json_submit: Path,
    log_base: Path,
    log_dir: Path
) -> str:
    """
    Processes GEV fits from event data.

    Args:
        event_file (Path): Path to the events file.
        gev_dir (Path): Directory for GEV output.
        gev_args (List[str]): Additional arguments for GEV processing.
        extra_args (List[str]): Extra arguments for processing.
        hold_after (Optional[str]): Dependencies for the job to hold after.
        log_file (str): Log file name.
        job_name (str): Job name.
        json_submit (Path): Path to the JSON submission configuration.
        log_base (Path): Base directory for logs.
        log_dir (Path): Directory for the log file.

    Returns:
        str: The holdafter string for the submitted job.
    """
    cmd: List[str] = [
        "process_gev_fits.py",
        str(event_file),
        "--outdir", str(gev_dir),
        "--nsamples=100"
    ] + gev_args + extra_args
    return submit_job(cmd, log_file, job_name, hold_after, json_submit, log_base, log_dir)

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process radar data.")
    parser.add_argument("summary_dir", type=Path, help="Directory of the radar data to process")
    parser.add_argument("--region", nargs=5, metavar=("x0", "y0", "x1", "y1", "reg_name"), help="Region to extract")
    parser.add_argument("--radius", type=float, help="Radius for seasonal mask")
    parser.add_argument("--holdafter", nargs="+", help="Jobs to hold after")
    parser.add_argument("--covariates", nargs="+", help="Covariates for GEV processing")
    parser.add_argument("--bootstrap", type=int, help="Number of bootstrap samples")
    args, extra_args = parser.parse_known_args()

    # Paths and variables
    name: str = args.summary_dir.name
    process_dir: Path = args.summary_dir.parent.parent / "processed" / name
    seas_str: str = "DJF"
    sm_file: Path = process_dir / f"seas_mean_{name}_{seas_str}.nc"
    nomask_file: Path = process_dir / f"seas_mean_{name}_{seas_str}_nomask.nc"
    event_file: Path = process_dir / f"events_seas_mean_{name}_{seas_str}.nc"
    gev_dir: Path = process_dir / "fits"
    pbs_log_dir: Path = process_dir / "pbs_logs"
    run_log_dir: Path = process_dir / "run_logs"
    time_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure directories exist
    pbs_log_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    jobs=[]

    # Process region
    region_args: List[str] = []
    hold_after: Optional[str] = None
    if args.region:
        x0, y0, x1, y1, reg_name = args.region
        region_args = ["--region", x0, y0, x1, y1]
        name += f"_{reg_name}"
        process_dir /= reg_name
        if not sm_file.exists():
            raise FileNotFoundError(f"Error: Seasonal mean file {sm_file} does not exist. Run process_seas_avg_mask.py first.")

    # Process steps
    if not args.region:
        hold_after = process_mean_mask(
            summary_dir=args.summary_dir,
            sm_file=sm_file,
            nomask_file=nomask_file,
            args=["--radius", str(args.radius)] if args.radius else [],
            extra_args=extra_args,
            hold_after=hold_after,
            log_file=f"process_seas_avg_mask_{name}_{time_str}",
            job_name=f"smn_{name}",
            json_submit=Path(AUSRAIN) / "process_seas_avg_mask.json",
            log_base=pbs_log_dir,
            log_dir=run_log_dir
        )
        jobs.append(hold_after)
    hold_after = process_events(
        seas_mean_file=sm_file,
        event_file=event_file,
        extra_args=extra_args,
        region_args=region_args,
        hold_after=hold_after,
        log_file=f"process_events_{name}_{time_str}",
        job_name=f"ev_{name}",
        json_submit=Path(AUSRAIN) / "process_events.json",
        log_base=pbs_log_dir,
        log_dir=run_log_dir
    )
    jobs.append(hold_after)
    hold_after = process_gev(
        event_file=event_file,
        gev_dir=gev_dir,
        gev_args=["--covariates"] + args.covariates if args.covariates else [],
        extra_args=extra_args,
        hold_after=hold_after,
        log_file=f"process_gev_fits_{name}_{time_str}",
        job_name=f"gev_{name}",
        json_submit=Path(AUSRAIN) / "process_gev_fits.json",
        log_base=pbs_log_dir,
        log_dir=run_log_dir
    )
    jobs.append(hold_after)

    print(f"Jobs submitted: {jobs}")