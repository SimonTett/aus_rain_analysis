# python code to plot masks.
# masks are beam blockage combined with land.
import ausLib
import xarray
import matplotlib.pyplot as plt
## get in data first
masks={}
topog={}
projs={}
regn={'x':slice(-125e3, 125e3), 'y':slice(-125e3, 125e3)}
for site,no in ausLib.site_numbers.items():

    files = list((ausLib.data_dir/f'site_data/{site}').glob(f"{site}_{no:03d}_*cbb_dem.nc"))
    print(site,'files are files',files)
    CBB_DEM = xarray.open_mfdataset(files, concat_dim='prechange_start', combine='nested').sel(**regn)
    # get the projection info -- use the first proj thing we find.
    for v in CBB_DEM.variables:
        if 'proj' in v:
            projs[site] = ausLib.radar_projection(CBB_DEM[v].attrs)
            break
    CBB_DEM = CBB_DEM.max('prechange_start').coarsen(x=4, y=4, boundary='trim').mean()
    bbf = CBB_DEM.CBB.clip(0, 1)
    # keep where BBF < 0.5 and land.
    msk = (bbf < 0.5) & (CBB_DEM.elevation > 0.0)
    masks[site]= msk
    topog[site] = CBB_DEM.elevation.where(msk)





## make plots
# different projections for each plot. So set it up!
per_subplot_kwagrs= {site:dict(projection=proj) for site,proj in projs.items()}
#fig,axes = ausLib.std_fig_axs('masks',per_subplot_kw=per_subplot_kwagrs)
fig,axes = ausLib.std_fig_axs('masks',add_projection=True)
levels=[0,50,100,150,200,300,400,500,600,700,800,900,1000,1500,2000,2500]
for site,ax in axes.items():
    #a.set_extent([-125e3,125e3,-125e3,125e3])
    cm=topog[site].plot(ax=ax, cmap='gist_earth', add_colorbar=False, levels=levels)#,vmin=0,vmax=2100))#
    ax.coastlines(color='purple',linewidth=1)
    ax.set_title(site)
    ax.plot(0.0,0.0,marker='h',color='red',markersize=10)
fig.colorbar(cm, ax=axes.values(), orientation='horizontal', label='Elevation (m)')
fig.show()

