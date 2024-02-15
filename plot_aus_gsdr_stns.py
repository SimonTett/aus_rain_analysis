# plot the Australian GSDR stations
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import cartopy.crs as ccrs
import seaborn as sns
import ausLib
import pandas as pd

## let's plot what we have but in the period we care about.
all_metadata = ausLib.read_gsdr_csv("AU_GSDR_metadata.csv")
L = (all_metadata.End_time >= '2015') & (all_metadata.Start_time <= '2000')
current = all_metadata[L]
# read in the actual values and compute the max_rain (for the entire record)
mx = dict()
for stn, record in current.iterrows():
    data = ausLib.read_gsdp_data(record)
    mx[stn] = data.max()
mx = pd.Series(mx).rename('max_rain')
current = pd.concat([current, mx], axis=1)
## actually plot the data

fig = plt.figure(clear=True, num='AU_GSDR', figsize=(8, 5), layout='constrained')
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

yr_durn = (current.End_time - current.Start_time).dt.total_seconds() / (365.24 * 24 * 60 * 60)
levels = [10, 15, 20, 30, 50, 75, 100, 125, 150, 200]
norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=len(levels) + 1, extend=True)  # does not work,
norm = matplotlib.colors.LogNorm(vmin=levels[0], vmax=levels[-1])
cmap = 'tab10'

cm = sns.scatterplot(current, x='Longitude', y='Latitude', marker='.', ax=ax, s=yr_durn * 4,
                     c=current.max_rain, norm=norm, cmap=cmap, legend='brief')
fig.colorbar(None, cmap=cmap, norm=norm, ax=ax, orientation='horizontal',
             fraction=0.1, aspect=40, pad=0.05, ticks=levels,
             format="%4d", spacing='uniform', label='Max Rain (mm/h)', shrink=0.7)
ax.set_title("GSDR Australia Stations 2000-2015")
ax.coastlines()
fig.show()
fig.savefig("AU_GSDR.png")
