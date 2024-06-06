# extract all meta-data from GSDR data.
# then write to a csv file. Note that the metadata is extracted from the individual records.
# Times are converted to UTC and some other changes are made.
import pathlib
import ausLib

my_logger = ausLib.my_logger
ausLib.init_log(my_logger, level='DEBUG')


root_dir = pathlib.Path("/g/data/ua8/Precipitation/GSDR")
files_5min=list((root_dir/"aus5min_postqc").glob("AU_*.txt"))
files_1min=list((root_dir/"aus1min_postqc").glob("AU_*.txt"))
all_files = files_5min+files_1min

my_logger.info(f"Processing {len(all_files)} files ")
all_metadata = ausLib.gsdr_metadata(all_files)
# having generated it let's save it.
all_metadata.to_csv("meta_data/AU_GSDR_metadata.csv")
