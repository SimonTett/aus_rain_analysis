# quick test -- can we read in data -- for process_extract_metadata.read_ppi

import ausLib
import xarray
import process_extract_metadata
import pathlib
import itertools
site='Sydney'
site_number = f'{ausLib.site_numbers[site]:d}'
indir = pathlib.Path(f'/g/data/rq0/level_1b/{site_number}/ppi')

year=2000

pattern = f'{site_number}_{year:04d}*_ppi.zip'
data_dir = indir / f'{year:04d}'
zip_files = sorted(data_dir.glob(pattern))  #, key=month_sort)

# these are all daily files. But we want the first one from each month.
files = []
level_1_files=[]
ukeys = []
for k, g in itertools.groupby(zip_files, key=process_extract_metadata.month_sort):
    file = list(g)[0]
    files.append(file)  # just want the first file from each month.
    parts = file.parts
    lev1=list(parts[0:4])+['level_1','odim_pvol',parts[5],parts[7],'vol']
    lev1 += [parts[-1].replace('_ppi.zip', '.pvol.zip')]
    lev1=pathlib.Path(*lev1)
    if not lev1.exists():
        lev1=None

    level_1_files.append(lev1)
    ukeys.append(k)

ds=process_extract_metadata.read_ppi(files[0],level1_file=level_1_files[0])