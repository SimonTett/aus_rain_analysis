# count hte number of tropical cylones nearby radar.

## see how many TC dates we have.
import cartopy.geodesic
import numpy as np
import ausLib
import xarray

g = cartopy.geodesic.Geodesic()
tracks = xarray.load_dataset(ausLib.common_data / 'IBTrACS.since1980.v04r01.nc')

for site, no in ausLib.site_numbers.items():
    info = ausLib.site_info(no)
    dist = g.inverse(info.iloc[0].loc[['site_lon', 'site_lat']].astype('float').values,
                     np.array([tracks.lon, tracks.lat]).T
                     )
    L = dist[:, 0] < 500e3  # within 500 km
    breakpoint()
    L = L.reshape(tracks.lon.shape)
    coords = [tracks.lon.storm, tracks.lon.date_time]
    L = xarray.DataArray(L, coords=coords)

    # dates
    tc_dates = np.unique(np.datetime_as_string(tracks.time.values.flatten()[L],
                                               unit='D'
                                               )
                         )[:-1]

    # and see how many dates of max we have.
    match = time_extremes[site].dt.strftime('%Y-%m-%d').isin(tc_dates.tolist())
    no = match.sum().values
    if no == 0:
        print(f'{site}: No tc dates ')
    else:
        print(f'{site}: {no} tc dates')
        # print out the unique dates
        udates = np.unique(time_extremes[site].dt.strftime('%Y-%m-%d').values[match])
        print(site, udates)
        # and the names
        L2 = tracks[L].time.dt.strftime('%Y-%m-%d').isin(udates)
        print(site, np.unique(tracks.name[L2]))