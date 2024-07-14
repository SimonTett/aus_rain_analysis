# code to try and figure out why new processing leads to different gev fit from old processing
import pathlib
import xarray
import numpy as np
import matplotlib.pyplot as plt
import ausLib


def proc_events(file, threshold=0.5):
    radar_dataset = xarray.load_dataset(file, drop_variables=['xpos', 'ypos',
                                                              'fraction', 'sample_resolution',
                                                              'height',  'time'])
    radar_dataset = radar_dataset.sel(resample_prd=['30min', '1h', '2h'])
    # convert radar_dataset to accumulations.

    msk = (radar_dataset.max_value > threshold)
    radar_msk = radar_dataset.where(msk)
    mn_temp = radar_msk.ObsT.mean(['quantv', 'EventTime'])
    radar_msk['Tanom'] = radar_msk.ObsT - mn_temp  #
    return radar_msk


new = proc_events(
    ausLib.data_dir / 'processed/Sydney_rain_melbourne_check/events_seas_mean_Sydney_rain_melbourne_check_DJF.nc')
orig = proc_events(ausLib.data_dir / 'processed/Sydney_rain_melbourne/events_Sydney_rain_melbourne_DJF.nc')

# check times, max_value and obs_t for quantv=0.5
sel = dict(resample_prd='1h', quantv=0.5)
fig, axs = plt.subplots(1, 3, clear=True, num='test_event_gev_changes', figsize=(8, 6),
                        layout='constrained')
for ax, var in zip(axs, ['max_value', 't', 'ObsT']):
    ax.scatter(new[var].sel(**sel), orig[var].sel(**sel))
    ax.set_xlabel('new')
    ax.set_ylabel('orig')
    ax.set_title(var)

fig.show()

# problem is events
