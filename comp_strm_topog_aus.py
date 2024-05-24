# glue together multiple SRTM patches.  Consumes lots of memory...
# reduce resolution to 90m as intended purpose is to calculate beam blockage at ~ 1km resoln.
import rioxarray
import rioxarray.merge
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import ausLib

STRM_dir =pathlib.Path('/home/z3542688/data/aus_rain_analysis/SRTM_Data/')
files = list(STRM_dir.glob('*.hgt'))
ds_raw = rioxarray.merge.merge_arrays([rioxarray.open_rasterio(f,parallel=True) for f in files])
# and select within +/- 2 degrees of Melbourne (roughly +/- 150 km) so we don't run out of memory.
melbourne = dict(x=144.775,y=-37.85)
ds_raw = ds_raw.sel(x=slice(melbourne['x']-2,melbourne['x']+2.),y=slice(melbourne['y']+2.,melbourne['y']-2.))
ds_raw = ds_raw.where(ds_raw != ds_raw._FillValue,0.0 ) # Set missing values to zero.

# replace missing values with 0.0
# make approx 90km, 1km & 2km resolution dataset and save those.
# Note that STRM data is on long/lat grid while radar data is on "local" grid. Sigh!
print("opened STRM dataset",ausLib.memory_use())
ds_res=dict()
for resoln in [90,1000.,2000.]:
    coarse= np.round(resoln/30).astype(int)
    filename = STRM_dir/f'srtm_melbourne_approx_{int(resoln):d}m.tif'
    ds = ds_raw.coarsen(x=coarse,y=coarse,boundary='pad').mean().load()
    ds.rio.to_raster(filename)
    ds_res[resoln]= ds.squeeze().drop_vars('band')
    print(filename,coarse,ausLib.memory_use())

## plot all three resolutions
proj = ccrs.TransverseMercator(*melbourne.values())
fig,axs = plt.subplots(nrows=1,ncols=len(ds_res),figsize=(9,5),num='STRM DEM',clear=True,
                       subplot_kw=dict(projection=proj),layout='constrained')
levels = [-200,0.1,100,200,300,400,500,600,700,800,900,1000,1500]
for (res,d),ax in zip(ds_res.items(),axs):
    cm=d.plot(cmap='terrain',ax=ax,transform=ccrs.PlateCarree(),levels=levels,add_colorbar=False)
    ax.set_title(f'Approx {res}m')
    ax.coastlines()
fig.colorbar(cm,ax=axs,orientation='horizontal',fraction=0.05,pad=0.04)
fig.suptitle('SRTM data at different resolutions')
fig.show()

