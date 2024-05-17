# glue together multiple SRTM patches.  Consumes lots of memory...
# reduce resolution to 90m as intended purpose is to calculate beam blockage at ~ 1km resoln.
import rioxarray
import rioxarray.merge
import pathlib
import matplotlib.pyplot as plt
STRM_dir =pathlib.Path('/home/z3542688/data/aus_rain_analysis/SRTM_Data/')
files = list(STRM_dir.glob('*.hgt'))
ds = rioxarray.merge.merge_arrays([rioxarray.open_rasterio(f).coarsen(x=3,y=3,boundary='pad').mean() for f in files])
ds.rio.to_raster(STRM_dir/'srtm_australia_90m.tif')
melbourne = dict(y=-37.814167, x=144.963056)
ds.sel(x=slice(melbourne['x']-0.5,melbourne['x']+0.5),y=slice(melbourne['y']+0.5,melbourne['y']-0.5)).plot()