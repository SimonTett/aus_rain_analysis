# extract all meta-data from GSDR data.
# then write to a csv file
import pathlib

import pandas as pd

import ausLib
import logging


root_dir = pathlib.Path("/g/data/ua8/Precipitation/GSDR")
files_5min=list((root_dir/"aus5min_postqc").glob("AU_*.txt"))
files_1min=list((root_dir/"aus1min_postqc").glob("AU_*.txt"))
all_files = files_5min+files_1min
logging.basicConfig(level='INFO',force=True)
logging.info(f"Processing {len(all_files)} files ")
all_metadata = ausLib.gsdp_metadata(all_files)
# having generated it let's save it.
all_metadata.to_csv("AU_GSDR_metadata.csv")
