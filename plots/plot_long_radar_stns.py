# plot the long radar staions
import matplotlib.patches

import ausLib
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import commonLib

long_radar_data = ausLib.read_radar_file("meta_data/long_radar_stns.csv")
all_radar_data = ausLib.read_radar_file("meta_data/radar_site_list.csv")
fig = plt.figure(num="long_radar_stns", figsize=(8, 5), clear=True, layout='constrained')
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
sns.scatterplot(data=all_radar, x='site_lon', y='site_lat', ax=ax, sizes=8, marker='*',legend=False, color='grey')
sns.scatterplot(data=lr, x='site_lon', y='site_lat', hue='Type', style='Beamwidth',
                ax=ax, sizes=(40, 90), size='Changes', markers=['o', 's', 'h'], legend='full')
ax.legend(ncol=3,fontsize='small',handletextpad=0.1,handlelength=0.5)
# add 125x125 km boxes centred on the radars.
#lr.plot.scatter(x='site_lon',y='site_lat',marker='o',ax=ax,s=20)

dx=125e3
for name, row in lr.iterrows():
    n = ausLib.site_names[row.id]
    ax.text(row.site_lon, row.site_lat, n, va='bottom',size='small')
    albers_projection = ccrs.AlbersEqualArea(central_longitude=row.site_lon, central_latitude=row.site_lat)
    rect_patch=matplotlib.patches.Rectangle((-dx,-dx),2*dx,2*dx,linewidth=1,edgecolor='r',facecolor='none',transform=albers_projection)
    ax.add_patch(rect_patch)

# add on lines for regions
text_kwargs=dict(va='center',ha='left',color='royalblue',backgroundcolor='lightgrey',fontweight='bold',zorder=-5)
ax.hlines(-21,137,155,color='k',linestyle='--')
ax.hlines(-30,137,155,color='k',linestyle='--')
ax.vlines(147,-45,-21,color='k',linestyle='--')
ax.vlines(137,-37,-30,color='k',linestyle='--')
ax.vlines(137,-21,-10.5,color='k',linestyle='--')
ax.text(139,-19.5,'Tropics',**text_kwargs)
ax.text(155,-25.25,'QLD',**text_kwargs)
ax.text(155,-34.25,'NSW',**text_kwargs)
ax.text(141,-32,'South',**text_kwargs)
# add 35S line in blue -- shows the southern limit of TRMM  nadir views
# Once understand how TRMM being used can show southern limit of data..
ax.axhline(-35.0,color='blue',linestyle='--')
ax.set_title("Australian Radars")
fig.show()
commonLib.saveFig(fig)
