# plot GSDR stns with in 150 km of the Grafton radar.
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import cartopy.crs as ccrs
import cartopy.geodesic
import seaborn as sns
import ausLib
import pandas as pd
import xarray
import numpy as np
import cartopy.feature as cfeature
all_metadata = ausLib.read_gsdr_csv("AU_GSDR_metadata.csv")
L = (all_metadata.End_time >= '2015') & (all_metadata.Start_time <= '2000')
current = all_metadata[L]
# want at least 10 years of data ??
yr_durn = (current.End_time - current.Start_time).dt.total_seconds() / (365.24 * 24 * 60 * 60)
current = current[yr_durn > 10]
ds = xarray.open_dataset(ausLib.radar_dir/"28/RAINRATE/28_19980923_rainrate.nc")
centre = [ds.attrs['origin_longitude'],ds.attrs['origin_latitude']]
earth_geo = cartopy.geodesic.Geodesic()
pts = np.array([current.Longitude,current.Latitude]).T
dist = earth_geo.inverse(centre,pts)
L = dist[:,0] < 150e3 # closer than 150 km
current = current[L] # extract to 150 km radius,

mx = dict()
fract=dict()
for stn, record in current.iterrows():
    data = ausLib.read_gsdp_data(record).loc['1999-01-01':]
    fract[stn]=data.count() / data.size
    mx[stn] = data.max()
mx = pd.Series(mx).rename('max_rain')
fract = pd.Series(fract).rename('Fract_OK')
current = pd.concat([current, mx,fract], axis=1)
L=current.Fract_OK > 0.8
current_OK=current[L]
states = cfeature.NaturalEarthFeature(category='cultural', scale='10m',
                                      name='admin_1_states_provinces_lines',facecolor='none',edgecolor='red',linewidth=2)
## actually plot the data

fig = plt.figure(clear=True, num='AU_GSDR_grafton', figsize=(8, 5), layout='constrained')
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.plot(centre[0],centre[1],marker='x',ms=20)
levels = np.linspace(40,100,13)
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=len(levels) + 1, extend=True)  # does not work,
norm = matplotlib.colors.Normalize(vmin=levels[0], vmax=levels[-1])

cmap = 'tab10'
m = sns.scatterplot(current, x='Longitude', y='Latitude', marker='+', ax=ax, s=80,
                     c=current.max_rain, norm=norm, cmap=cmap, legend='brief')
cm = sns.scatterplot(current_OK, x='Longitude', y='Latitude', marker='o', ax=ax, s=80,
                     c=current_OK.max_rain, norm=norm, cmap=cmap, legend='brief')
fig.colorbar(None, cmap=cmap, norm=norm, ax=ax, orientation='vertical',
             fraction=0.1, aspect=40, pad=0.05, ticks=levels,
             format="%4d", spacing='uniform', label='Max Rain (mm/h)', shrink=0.7)
ax.set_title("GSDR Grafton Stations 2000-2015")
ax.coastlines()
ax.add_feature(states)
fig.show()
fig.savefig("AU_GSDR_grafton.png")

