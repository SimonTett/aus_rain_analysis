# plot radar values for extreme rain from Melbourne radar
# extreme happened on 2020-12-22
import matplotlib as mpl
import matplotlib.patches as mpatches
import ausLib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import commonLib
import copy

proj = ccrs.TransverseMercator(144.7554, -37.8553)
ds = ausLib.read_radar_zipfile(ausLib.data_dir / '2_20201222.gndrefl.zip', concat_dim='valid_time', combine='nested')
ds['x'] = ds['x'] * 1000
ds['y'] = ds['y'] * 1000
tgt = ds.reflectivity.isel(valid_time=254)  # When extremes happen.

## plot data
reg = dict(x=slice(-10e3, 0), y=slice(37e3, 29e3))
cmap = mpl.colormaps['RdBu'].with_extremes(under='white', over='black')
circ = mpatches.Circle(xy=(-5e3, 33e3), radius=3e3, color='black', transform=proj, fill=False, linewidth=2)
fig, axs = plt.subplots(nrows=2, ncols=2, num='radar_extreme',
                        clear=True, layout='constrained', subplot_kw=dict(projection=proj), sharex='col', sharey='col'
                        )
cm = tgt.where(tgt > 15).plot(ax=axs[0][0], levels=[20, 30, 40, 50, 60, 70], cmap=cmap,
                              x='x', y='y', transform=proj, add_colorbar=False
                              )
tgt.where(tgt > 15).sel(**reg).plot(ax=axs[0][1], levels=[20, 30, 40, 50, 60, 70], add_colorbar=False,
                                    cmap=cmap, x='x', y='y', transform=proj
                                    )
# now coarsen
tgtc = tgt.coarsen(x=4, y=4, boundary='trim').median()
cmc = tgtc.where(tgtc > 15).plot(ax=axs[1][0], levels=[20, 30, 40, 50, 60, 70], cmap=cmap,
                                 x='x', y='y', transform=proj, add_colorbar=False
                                 )
tgtc.where(tgtc > 15).sel(**reg).plot(ax=axs[1][1], levels=[20, 30, 40, 50, 60, 70], add_colorbar=False,
                                      cmap=cmap, x='x', y='y', transform=proj
                                      )
for ax in axs.flatten():
    circle = ax.add_patch(copy.deepcopy(circ))
    #ax.scatter(high_values.x.values,high_values.y.values,marker='o',color='black',s=50,transform=proj)
    ax.coastlines()
fig.colorbar(cm, ax=axs, orientation='horizontal', fraction=0.05, pad=0.04)
fig.show()
# and plot zoom in
commonLib.saveFig(fig)
