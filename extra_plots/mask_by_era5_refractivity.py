"""
Trial code to mask radar data by ERA-5 reflectivty data.
Will do this for a day at a time. Requires interpolating ERA-5 data to radar data.
Mask condition (at site) is:
(dctb >= 0.0 ) & (dctb < 750.) & (dndzn < -0.157)
>=0 means not missing. Want close enough to ground and want ducting
"""
import pathlib
import typing

import cartopy.crs as ccrs
import xarray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

import ausLib


def era5_files(time:pd.Timestamp) -> pathlib.Path|list[pathlib.Path]:
    """
    Return era5 file or list of files that match the desired time.

    Args:
        time: time wanted. Used to select files.
        Exact behaviour depends on where is being run

    """

    if ausLib.platform == 'geos':
        gpattern = f"era5_refractivity_{time.year}_*.nc"
        era_file_list = ausLib.era5_dir.glob(gpattern)
        if not era_file_list:
            raise ValueError(f" {ausLib.era5_dir}.glob({gpattern}) failed to find any files.")

    else:
        # find list of files. gadi keeps them for each year/month and grouped by var.
        vars=['tplt','tplb','dctb','dndza','dndzn'] # all vars we need
        era_file_list = []
        for var in vars:
            dir = ausLib.era5_dir/f'single-levels/reanalysis/{var}/{time.year}'
            files = list(dir.glob(f'{var}_era5_oper_sfc_{time.year}{time.month:02d}01*.nc'))
            era_file_list += files
            logging.info(f"var: {var} files:{files}")


    return era_file_list




_SAVE_SENTINAL=  object()

parser = argparse.ArgumentParser(description='Mask radar data by ERA-5 reflectivty data and make plot.')
parser.add_argument('station',choices=ausLib.site_numbers.keys(),help='Radar to use.')
parser.add_argument('time', type=pd.Timestamp, help='Time to mask radar data and plot.')
parser.add_argument('--save',nargs='?', type=pathlib.Path, help='save file.',
                    const=_SAVE_SENTINAL, default=None,metavar="FILENAME")
args = parser.parse_args()
time: pd.Timestamp = args.time
site_number = ausLib.site_numbers[args.station]

save_file = args.save
if save_file is _SAVE_SENTINAL:
    save_file=pathlib.Path(f"{args.station}_{time.strftime('%Y_%m_%d_%H%M')}.png")
radar_file = ausLib.hist_ref_dir / f'{site_number:02d}/{time.year}/{site_number:02d}_{time.strftime("%Y%m%d")}.gndrefl.zip'
if not radar_file.exists():
    raise ValueError(f"{radar_file} not found.")
## read in the data
## read in radar reflectivity
radar_data = ausLib.read_radar_zipfile(radar_file, drop_vars=['error', 'x_bounds', 'y_bounds'])
if radar_data is None:
    raise ValueError(f"Problem reading in data from {radar_file}")
# convert from km to m
for k in ['x', 'y']:
    radar_data[k] = radar_data[k] * 1000.
    radar_data[k].attrs['units'] = 'm'

proj = ausLib.radar_projection(radar_data.proj.attrs)
radar_data = ausLib.add_long_lat_coords(radar_data)
reflectivity = radar_data.reflectivity
radar_stn_coords = {k.split("_")[0]: radar_data.proj.attrs[k] for k in radar_data.proj.attrs if k.startswith(('latitude', 'longitude'))}

## then era-5 refractivty info
era_files = era5_files(time)
long_stn = tuple(radar_stn_coords['longitude'] + d for d in (-2, 2))
lat_stn = tuple(radar_stn_coords['latitude'] + d for d in (4, -4))
era = xarray.open_mfdataset(era_files).sel(longitude=slice(*long_stn), latitude=slice(*lat_stn)).rename(time='valid_time')
era_ref = era.sel(valid_time=time.strftime("%Y-%m-%d"))

# mask where era_ref.tplt  & tplb >0 #
era_ref=era_ref.where((era_ref.tplt >= 0.0) & (era_ref.tplb >=0))
# nearest neighbour regrid era-5 to reflectivity data.
era5_ref_radar = era_ref.interp(longitude=reflectivity.longitude,
                                latitude=reflectivity.latitude,
                                valid_time = (reflectivity.valid_time - pd.Timedelta("30min")).values,
                                method="nearest")

## now plot the reflectivity and its masked value.
kwrds_cbar = dict(orientation='horizontal',)
kwrds_cbar = dict()
plot_kwrds = dict(
    dndza = dict(levels=[-0.150,-0.079,0.079,0.150]),
    dndzn = dict(levels=[-0.150, -0.079, 0.079,0.150]),
    tplb= dict(vmin=0,vmax=3000,levels=[0,10,20,50,100,200,500,1000,2000]),
    dctb = dict(levels=[0,10,20,50,100,200,500,1000,2000],vmin=0,vmax=3000),
    tplt= dict(vmin=0,vmax=3000,levels=[0,10,20,50,100,200,500,1000,2000]   ),
    duct_thick= dict(levels=[0,10,20,50,100,200,500,1000,2000],vmin=0,vmax=2000)    )
fig,axs  = plt.subplots(2,2,sharex=True,sharey=True,
                                     clear=True,num='masked_ref',figsize=(10,10),subplot_kw=dict(projection=proj))

ax_ref=axs.flat[0]
ref = reflectivity.sel(valid_time=time, method='nearest')
era5_time = era5_ref_radar.sel(valid_time=time,method='nearest').where(ref.notnull())
era5_time = era5_time.where(era5_time.tplt >= 0.0)
era5_time['duct_thick'] = era5_time.tplt - era5_time.dctb
ref = ref.where(ref > 0) # #
kwrds=dict(ax=ax_ref,transform=proj,vmin=0,vmax=60,cmap='RdBu',cbar_kwargs =kwrds_cbar)
ref.plot(**kwrds)
min_refract = era5_time.dndzn
msk = (min_refract < -0.079).astype(float).where(~min_refract.isnull())
msk = (era5_time.tplb < 50.).astype(float)
msk.plot(alpha=0.25,add_colorbar=False,cmap='binary',ax=ax_ref,vmin=0,vmax=1,levels=[0.5])


ax_ref.set_title(f'Reflectivity')
for ax,var in zip(axs.flat[1:],['dndzn','dctb','duct_thick']):
    da = era5_time[var]
    kw_plt =plot_kwrds.get(var,{})
    #print(var,kw_plt,int(da.notnull().sum()))
    da.plot(ax=ax,cmap='RdBu',cbar_kwargs =kwrds_cbar,**kw_plt) # ,
    ax.set_title(f'{var} ')
for axs in axs.flat:
    axs.set_extent((-125e3, 125e3, -125e3, 125e3), crs=proj)
    axs.plot(138.5024,-35.3295,marker='x',color='black',markersize=10,transform=ccrs.PlateCarree())
    axs.coastlines()
fig.suptitle(ref.valid_time.dt.strftime("%Y-%m-%d:%H%M").values)
fig.show()
if save_file:
    fig.savefig(save_file,dpi=300)
