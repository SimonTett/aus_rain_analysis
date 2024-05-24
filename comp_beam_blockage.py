# Compute beam blockage factor. Based off https://docs.wradlib.org/en/stable/notebooks/beamblockage/beamblockage.html
# see https://hess.copernicus.org/articles/17/863/2013/hess-17-863-2013.pdf (and cite it!)
from typing import Tuple

import wradlib as wrl
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray
import xradar
import warnings
import osgeo
import numpy as np
import pathlib
import rioxarray
import cartopy.crs as ccrs
import ausLib
# for sites apart from Mebourne there are (small) changes in ht/location and beamwidth. Will need to deal with that.
# fortunately forme Melbourne is "easy"
STRM_dir = ausLib.data_dir/'SRTM_Data'
sitecoords: tuple[float, float, float] = (144.7555, -37.8553, 45.)  # from stn metadata file.
site = 'Melbourne'
outfile = ausLib.data_dir/f'ancil/{site}_cbb_dem.nc'

outfile.parent.mkdir(exist_ok=True, parents=True)
nrays = 360  # number of rays
nbins = 400  # number of range bins
el = 0.5  # vertical antenna pointing angle (deg)
bw = 1.0  # half power beam width (deg)
range_res = 500.0  # range resolution (meters)
r = np.arange(nbins) * range_res
beamradius = wrl.util.half_power_radius(r, bw)
coord = wrl.georef.sweep_centroids(nrays, range_res, nbins, el)
coords = wrl.georef.spherical_to_proj(
    coord[..., 0], coord[..., 1], coord[..., 2], sitecoords
)
lon = coords[..., 0]
lat = coords[..., 1]
alt = coords[..., 2]
polcoords = coords[..., :2]
print("lon,lat,alt:", coords.shape)
rlimits = (lon.min(), lat.min(), lon.max(), lat.max())
print(
    "Radar bounding box:\n\t%.2f\n%.2f             %.2f\n\t%.2f"
    % (lat.max(), lon.min(), lon.max(), lat.min())
)

rasterfile = STRM_dir / "srtm_melbourne_approx_90m.tif"

ds = rioxarray.open_rasterio(rasterfile).sel(x=slice(rlimits[0], rlimits[2]), y=slice(rlimits[3], rlimits[1]))
rastervalues = ds.values.squeeze()
rastercoords = np.stack(np.meshgrid(ds.x, ds.y), axis=-1)
crs = ds.rio.crs

# Map rastervalues to polar grid points. Q ? Where does the cord ref system come in?
polarvalues = wrl.ipol.cart_to_irregular_spline(
    rastercoords, rastervalues, polcoords, order=3, prefilter=False
)
proj_rad = wrl.georef.create_osr("aeqd", **dict(lat_0=sitecoords[1], lon_0=sitecoords[0], x_0=0., y_0=0.))
DEM = wrl.georef.create_xarray_dataarray(
    polarvalues, r=r, phi=coord[:, 0, 1], site=sitecoords
).wrl.georef.georeference(crs=proj_rad)

PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
PBB = np.ma.masked_invalid(PBB)
CBB = wrl.qual.cum_beam_block_frac(PBB)
CBB = wrl.georef.create_xarray_dataarray(
    CBB, r=r, phi=coord[:, 0, 1], site=sitecoords, theta=el,
).wrl.georef.georeference(crs=proj_rad)
# the cumulative beam blockage.
CBB.to_netcdf(STRM_dir / 'cbb_melbourne.nc')
# and regrid -- coords from inspection of the Melbourne ground reflectivity.
cart = xarray.Dataset(coords={"x": (["x"], np.arange(-127.75e3, 127.5e3, 500)),
                              "y": (["y"], np.arange(-127.75e3, 127.5e3, 500))}
                      )
src = np.stack([CBB.x.values.flatten(), CBB.y.values.flatten()], axis=-1)
trg = np.meshgrid(cart.x.values, cart.y.values)
trg = np.vstack((trg[0].ravel(), trg[1].ravel())).T
interpol = wrl.ipol.OrdinaryKriging # default krieging.
#interpol = wrl.ipol.OrdinaryKriging(src,trg)

CBB_grid = CBB.wrl.comp.togrid(cart, radius=200e3, center=(0, 0), interpol=interpol)
DEM_grid = DEM.wrl.comp.togrid(cart, radius=200e3, center=(0, 0), interpol=interpol)
ds_grid = xarray.Dataset(dict(elevation=DEM_grid,CBB=CBB_grid))
ds_grid.to_netcdf(outfile) # and save it

## now to plot things.
# just a little helper function to style x and y axes of our maps
def annotate_map(ax, cm=None, title=""):

    if cm is not None:
        plt.colorbar(cm, ax=ax,orientation='horizontal',fraction=0.05, pad=0.04)
    if not title == "":
        ax.set_title(title)
    ax.coastlines(resolution="10m", color="black", linewidth=1)
    #ax.gridlines(draw_labels=True)


# plot DEM, blockage and down-range
fig = plt.figure(figsize=(8, 7), clear=True, num='Radar Blockage',layout='constrained')

# create subplots

proj = ccrs.TransverseMercator(*sitecoords[0:2])  # centred on the radar
ax1 = plt.subplot2grid((2, 3), (0, 0), projection=proj)
ax2 = plt.subplot2grid((2, 3), (0, 1), projection=proj)
ax_grid = plt.subplot2grid((2, 3), (0, 2), projection=proj)
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=3, rowspan=1)

# azimuth angle
angle = 330

# Plot terrain (on ax1)

dem = DEM.plot(ax=ax1, cmap=mpl.cm.terrain, x='x', y='y', transform=proj,
               vmin=0.0, add_colorbar=False, robust=True
               )
ax1.plot(
    [0, np.sin(np.radians(angle)) * 150e3],
    [0, np.cos(np.radians(angle)) * 150e3], "r-", transform=proj
)
ax1.plot(sitecoords[0], sitecoords[1], "ro", transform=ccrs.PlateCarree())
annotate_map(ax1, dem, title="Terrain < {0} km range".format(np.max(r / 1000.0) + 0.1))

# Plot CBB (on ax2)

cbb = CBB.plot(ax=ax2, cmap=mpl.cm.PuRd, vmin=0, vmax=1, add_colorbar=False,
               x='x', y='y', transform=proj
               )
annotate_map(ax2,  title="BBF")
ax2.plot(sitecoords[0], sitecoords[1], "ro", transform=ccrs.PlateCarree())
# plot gridded
cbb_grid = CBB_grid.plot(ax=ax_grid, cmap=mpl.cm.PuRd, vmin=0, vmax=1, add_colorbar=False,
                         x='x', y='y', transform=proj)
annotate_map(ax_grid, cbb_grid, "BBF (gridded)")
# Plot single ray terrain profile on ax3
(bc,) = ax3.plot(r / 1000.0, alt[angle, :], "-b", linewidth=3, label="Beam Center")
(b3db,) = ax3.plot(
    r / 1000.0,
    (alt[angle, :] + beamradius),
    ":b",
    linewidth=1.5,
    label="3 dB Beam width",
)
ax3.plot(r / 1000.0, (alt[angle, :] - beamradius), ":b")
ax3.fill_between(r / 1000.0, 0.0, DEM.sel(azimuth=angle, method='nearest'), color="0.75")
ax3.set_xlim(0.0, np.max(r / 1000.0) + 0.1)
ax3.set_ylim(0.0, 3000)
ax3.set_xlabel("Range (km)")
ax3.set_ylabel("Altitude (m)")
ax3.grid()

axb = ax3.twinx()
(bbf,) = axb.plot(r / 1000.0, CBB[angle, :], "-k", label="BBF")
axb.set_ylabel("Beam-blockage fraction")
axb.set_ylim(0.0, 1.0)
axb.set_xlim(0.0, np.max(r / 1000.0) + 0.1)

legend = ax3.legend(
    (bc, b3db, bbf),
    ("Beam Center", "3 dB Beam width", "BBF"),
    loc="upper left",
    fontsize=10,
)

fig.show()


## plot beam along with Earth Curvature
def height_formatter(x, pos):
    x = (x - 6370000) / 1000
    fmt_str = "{:g}".format(x)
    return fmt_str


def range_formatter(x, pos):
    x = x / 1000.0
    fmt_str = "{:g}".format(x)
    return fmt_str


fig = plt.figure(figsize=(8, 5),clear=True, num='radar_beam_curvature', layout='constrained')

cgax, caax, paax = wrl.vis.create_cg(fig=fig, rot=0, scale=1)

# azimuth angle
angle = 330

# fix grid_helper
er = 6370000
gh = cgax.get_grid_helper()
gh.grid_finder.grid_locator2._nbins = 80
gh.grid_finder.grid_locator2._steps = [1, 2, 4, 5, 10]

# calculate beam_height and arc_distance for ke=1
# means line of sight
bhe = wrl.georef.bin_altitude(r, 0, sitecoords[2], re=er, ke=1.0)
ade = wrl.georef.bin_distance(r, 0, sitecoords[2], re=er, ke=1.0)
nn0 = np.zeros_like(r)
# for nice plotting we assume earth_radius = 6370000 m
ecp = nn0 + er
# theta (arc_distance sector angle)
thetap = -np.degrees(ade / er) + 90.0

# zero degree elevation with standard refraction
bh0 = wrl.georef.bin_altitude(r, 0, sitecoords[2], re=er)

# plot (ecp is earth surface normal null)
(bes,) = paax.plot(thetap, ecp, "-k", linewidth=3, label="Earth Surface NN")
(bc,) = paax.plot(thetap, ecp + alt[angle, :], "-b", linewidth=3, label="Beam Center")
(bc0r,) = paax.plot(
    thetap, ecp + bh0 + alt[angle, 0] - sitecoords[2], "-g", label="0 deg Refraction"
)
(bc0n,) = paax.plot(
    thetap, ecp + bhe + alt[angle, 0] - sitecoords[2], "-r", label="0 deg line of sight"
)
(b3db,) = paax.plot(
    thetap, ecp + alt[angle, :] + beamradius, ":b", label="+3 dB Beam width"
)
paax.plot(thetap, ecp + alt[angle, :] - beamradius, ":b", label="-3 dB Beam width")

# orography
paax.fill_between(thetap, ecp, ecp + DEM.sel(azimuth=angle, method='nearest'), color="0.75")

# shape axes
cgax.set_xlim(0, np.max(ade))
cgax.set_ylim([ecp.min() - 1000, ecp.max() + 2500])
caax.grid(True, axis="x")
cgax.grid(True, axis="y")
cgax.axis["top"].toggle(all=False)
caax.yaxis.set_major_locator(
    mpl.ticker.MaxNLocator(steps=[1, 2, 4, 5, 10], nbins=20, prune="both")
)
caax.xaxis.set_major_locator(mpl.ticker.MaxNLocator())
caax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(height_formatter))
caax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(range_formatter))

caax.set_xlabel("Range (km)")
caax.set_ylabel("Altitude (km)")

legend = paax.legend(
    (bes, bc0n, bc0r, bc, b3db),
    (
            "Earth Surface NN",
            "0 deg line of sight",
            "0 deg std refraction",
            "Beam Center",
            "3 dB Beam width",
    ),
    loc="upper left",
    fontsize=10,
)

fig.show()
