# compute the stations that match.
import ausLib
import pathlib
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import matplotlib.patches as patches

site = 'Melbourne'
range = 100e3
my_logger = ausLib.my_logger
ausLib.init_log(my_logger, 'DEBUG')

site_number = f'{ausLib.site_numbers[site]:d}'
indir = pathlib.Path('/g/data/rq0/hist_gndrefl') / (site_number+'/2010')
files = list(indir.glob('*gndrefl.zip'))
ds=ausLib.read_radar_zipfile(files[0],concat_dim='valid_time',parallel=True,
                      combine='nested',
                      chunks=dict(valid_time=6), engine='netcdf4',first_file=True)
proj=ausLib.radar_projection(ds.proj.attrs)
radar_coords = np.array([ds.proj.attrs['longitude_of_central_meridian'],
                         ds.proj.attrs['latitude_of_projection_origin']])
gauges_close = ausLib.read_gauge_metadata(radar_coords, range,
                                   time_range=('1997-01-01', '2022-12-31'))
my_logger.info(f'Have {gauges_close.shape[0]} gauges close to {site} radar')
# save the data
gauges_close.to_csv(f'meta_data/{site}_close.csv')
## make a map.

fig,ax = plt.subplots(nrows=1,ncols=1,num=f'{site}_close_gauges',figsize=(8,8),layout='constrained',
                      subplot_kw=dict(projection=ccrs.PlateCarree()),clear=True)
gauges_close.plot.scatter(x='Longitude',y='Latitude',transform=ccrs.PlateCarree(),marker='o',color='red',s=10,ax=ax)
ax.plot(0.0,0.0,transform=proj,marker='x',color='blue',ms=15)
ax.coastlines()
for rng in [25,50,75,100,125]:
    circle = patches.Circle((0.0,0.0),rng*1e3,edgecolor='k',facecolor='none',transform=proj)
    ax.add_patch(circle)

ax.set_title(f'Rain gauges close to {site} radar')
fig.show()