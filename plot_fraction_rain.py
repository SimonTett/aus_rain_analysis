# plot the 2021 fraction of missing/zero data.
# for  Sellick (46), Melbourne (id=2), West Takone (id=52), Sydney/Wollongong (id=3),  Brisbane (id=50) Cairns (id=19) ,Mornington Island (36),
import math
import pathlib
import cartopy.crs as ccrs
import xarray
import ausLib
import logging
import matplotlib.pyplot as plt
import multiprocessing

log = logging.getLogger("MAIN")
ausLib.init_log(log,level='INFO')

def comp_fract_rain(ds:xarray.Dataset) -> xarray.Dataset:
    """

    :param ds:
    :return:
    """
    time_dim='time'
    OK = ds.isfile == 1 # got some data.
    count = (ds.rainrate > 0.0).sum('time',min_count=1,skipna=True).load()
    log.info("Computed count ")
    nsamps = int(OK.sum('time').load())
    fract = (count/nsamps).rename("fraction").load()
    log.info("Computed fract ")
    coords = dict()
    for c in ['latitude', 'longitude']:
        try:  # potentially generating multiple values so just want first one
            coords[c] = ds[c].isel({time_dim: 0}).squeeze(drop=True).drop_vars(time_dim).load()
        except ValueError:  # no time so just have the coord
            coords[c] = ds[c].load()

    fract=fract.assign_coords(**coords)
    result = xarray.Dataset(dict(fraction=fract,nsamples=nsamps))
    return result
if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!
    ausLib.dask_client()
    radars=dict(Sellick=46,Melbourne=2,WTakone=52,Sydney=3,Brisbane=50,Cairns=19,MornIsland=36)
    year =2021
    fractions=dict()
    for name,id in radars.items():
        dir = ausLib.radar_dir/f"{id}/RAINRATE"
        files = dir.glob(f"{id}_{year}0[6-7]*.nc")
        dataset = xarray.open_mfdataset(files,parallel=True)
        log.info(f"loaded {dataset.attrs['instrument_name']}")

        fract = comp_fract_rain(dataset)
        fractions[name]=fract
        log.info(f"Done with {name} for {id}")


    ## now to plot the data
    nreg = len(fractions)
    sizes = math.ceil(math.sqrt(nreg))
    fig,axis = plt.subplots(ncols=sizes,nrows=sizes,squeeze=False,
                        num='fraction_present',figsize=(8,9),clear=True,layout='constrained',
                        subplot_kw=dict(projection=ccrs.PlateCarree()))
    levels=[0.005,0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.5]
    for (name,ds),ax in zip(fractions.items(),axis.flatten()):
        cm=ds.fraction.plot(ax=ax,x='longitude',y='latitude',cmap='RdYlBu',
                         add_colorbar=False,levels=levels,transform=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_title(name)
        ax.gridlines(draw_labels=True)

    fig.colorbar(cm,ax=axis,orientation='horizontal',spacing='uniform',extend='both',pad=0.075,aspect=40,fraction=0.075)
    fig.show()
    fig.savefig('figures/fraction.png',dpi=300)




