# compare radar and in situ monthly 1 hr extremes.

import ausLib
import xarray
import pathlib
import cartopy
import cartopy.geodesic
import numpy as np
radar_dir = pathlib.Path("/scratch/wq02/st7295/radar/")
radarname='newcastle'
pth = radar_dir/radarname/f"processed_{radarname}.nc"
radar = xarray.open_dataset(pth)

all_metadata = ausLib.read_gsdr_csv("AU_GSDR_metadata.csv")

yr_durn = (all_metadata.End_time - all_metadata.Start_time).dt.total_seconds() / (365.24 * 24 * 60 * 60)

all_metadata = all_metadata[yr_durn > 10]

centre = [radar.attrs['origin_longitude'],radar.attrs['origin_latitude']]
earth_geo = cartopy.geodesic.Geodesic()
pts = np.array([all_metadata.Longitude, all_metadata.Latitude]).T
dist = earth_geo.inverse(centre,pts)
L = dist[:,0] < 100e3 # closer than 150 km
radar_metadata = all_metadata[L] # extract to 150 km radius,