# plot the beam blockage Based off https://docs.wradlib.org/en/stable/notebooks/beamblockage/beamblockage.html
# # see https://hess.copernicus.org/articles/17/863/2013/hess-17-863-2013.pdf (and cite it!)


import wradlib as wrl
import matplotlib.pyplot as plt
import commonLib
import xarray

import ausLib

# load data.
data_dir = ausLib.data_dir / 'SRTM_Data/radar_strm'
site = 'Melbourne'
site_index = ausLib.site_numbers[site]
files = list(data_dir.glob(f'{site_index:03d}_[0-9]_*cbb_dem.nc'))
ds=xarray.open_mfdataset(files,concat_dim='prechange_start',combine='nested')
proj = ausLib.radar_projection(ds['proj_'+ds.attrs["long_id"]].attrs)
fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(8, 8),num=f'{site}_CBB',clear=True)
ds.CBB.max('prechange_start').plot(ax=ax, cmap='PuRd',center=False)
ax.coastlines(resolution="10m", color="black", linewidth=1)
ax.set_title(f'Max CBB for {site}')
fig.show()
commonLib.saveFig(fig)
