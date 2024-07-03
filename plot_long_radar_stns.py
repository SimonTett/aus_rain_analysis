# plot the long radar staions
import matplotlib.patches

import ausLib
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import commonLib

long_radar_data = ausLib.read_radar_file("meta_data/long_radar_stns.csv")
all_radar_data = ausLib.read_radar_file("meta_data/radar_site_list.csv")
fig = plt.figure(num="long_radar_aus", figsize=(8, 5), clear=True, layout='constrained')
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.set_extent([110, 160, -45, -10])
ax.coastlines()
lr = long_radar_data.groupby('id').tail(1)  # most recent records for each ID
lr = lr.sort_values('id', axis=0).set_index('id', drop=False)
# add in changes.
changes = long_radar_data.id.value_counts().rename("Changes")
lr = lr.join(changes).rename(columns=dict(radar_type='Type',beamwidth='Beamwidth'))
# plto all radars as tiny dots!
all_radar = all_radar_data.groupby('id').tail(1)
sns.scatterplot(data=all_radar, x='site_lon', y='site_lat', ax=ax, sizes=5, legend=False, color='grey')
sns.scatterplot(data=lr, x='site_lon', y='site_lat', hue='Type', style='Beamwidth',
                ax=ax, sizes=(40, 90), size='Changes', markers=['o', 's', 'h'], legend='full')
ax.legend(ncol=3,fontsize='small',handletextpad=0.1,handlelength=0.5)
# add 125x125 km boxes centred on the radars.
#lr.plot.scatter(x='site_lon',y='site_lat',marker='o',ax=ax,s=20)

dx=125e3
for name, row in lr.iterrows():
    n = ausLib.site_names[row.id]
    ax.text(row.site_lon, row.site_lat, n, va='bottom')
    albers_projection = ccrs.AlbersEqualArea(central_longitude=row.site_lon, central_latitude=row.site_lat)
    rect_patch=matplotlib.patches.Rectangle((-dx,-dx),2*dx,2*dx,linewidth=1,edgecolor='r',facecolor='none',transform=albers_projection)
    ax.add_patch(rect_patch)

ax.set_title("Australian radars.")
fig.show()
commonLib.saveFig(fig)
