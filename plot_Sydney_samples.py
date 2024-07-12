# plot the number of samples relative to expected samples for Sydney cases.
# This is to investigate if sub-sampling is doing what it should do.
import xarray
import ausLib
import matplotlib.pyplot as plt
import numpy as np
use_cache = True
my_logger = ausLib.setup_log(1)
if not (use_cache and 'max_samples' in locals()): # variables need to be computed.
    my_logger.info('Loading samples from cache')
    ds_base=xarray.open_mfdataset(((ausLib.data_dir/'summary/Sydney_rain_melbourne/').glob('*.nc')),parallel=True)
    ds_10min=xarray.open_mfdataset(((ausLib.data_dir/'summary/Sydney_rain_melbourne_10min/').glob('*.nc')),parallel=True)
    datasets={'base':ds_base,'10min':ds_10min}
    samples = {k:ds.count_raw_rain_rate.load() for k,ds in datasets.items()}
    max_samples = {k:(ds.time.dt.days_in_month*(np.timedelta64(1,'D')/ds.sample_resolution)).load()
               for k,ds in datasets.items()}
colors={'base':'blue','10min':'orange'}
#

## now to plot them
fig,ax=plt.subplots(1,1,figsize=(8,8),clear=True,num='Sydney_samples',layout='constrained')
for k,samp in samples.items():
    ratio = samp.resample(time='QS-DEC').sum()/max_samples[k].resample(time='QS-DEC').sum()
    ratio.plot(ax=ax,label=k,color=colors[k])
ax.axhline(0.7)
fig.show()
# biggest discrepcny si 2003-089 -- get that dat ain
ds_2003_09 = {k:ds.sel(time='2003-09').load() for k,ds in datasets.items()}

