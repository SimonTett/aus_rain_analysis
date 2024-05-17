# stuff to setup env
export AUSRAIN=~st7295/aus_rain_analysis
module load netcdf/4.7.3 # so have ncdump
module load cdo # cdo!
module load ncview
module load python3/3.11.7
module load openmpi/4.1.5
module load hdf5/1.12.2
module load R/4.3.1
module load python3-as-python # give me python 3!
module load gdal # needed for gdal stuff.
mpdule load nco # for nco
# and activate the virtual env
source $AUSRAIN/venv/bin/activate
# add in dask magic
module use /g/data/hh5/public/modules/
module load dask-optimiser

