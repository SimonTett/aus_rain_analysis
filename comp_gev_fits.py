# Compute the GEV fits
import numpy as np
import xarray
import ausLib
from R_python import gev_r
from numpy import random

def comp_radar_fit(dataset:xarray.Dataset,
                   n_samples:int=100,
                   rng_seed:int=123456) -> xarray.Dataset:
    rng = random.default_rng(rng_seed)
    rand_index = rng.integers(len(dataset.quantv), size=n_samples)
    ds = dataset.isel(quantv=rand_index).\
        rename(dict(quantv='sample')).assign_coords(sample=np.arange(0, 100))
    wt = ds.count_cells
    mx = ds.max_value

    fit = gev_r.xarray_gev(mx, dim='EventTime', weights=wt)
    return fit



site='Melbourne'
radar_dataset = xarray.load_dataset(ausLib.data_dir/f"events/{site}_hist_gndrefl_DJF.nc") # load the processed radar
# some of the max are actually no rain. So need to get rid of those.
threshold = 15. # min radar for actual rain. I guess in some seasons some locations have no rain!
msk = (radar_dataset.max_value > threshold ) & (radar_dataset.t.dt.year >= 2000) & (radar_dataset.max_value < 75)
radar_dataset = radar_dataset.where(msk)
rain = (10**(radar_dataset.max_value.astype('float64')/10.)/200.)**(5./8.)
#fit = gev_r.xarray_gev(radar_dataset.max_value,dim='EventTime')
fit = comp_radar_fit(radar_dataset)

