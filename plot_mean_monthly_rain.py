# plot the monthly-mean rain from radar
# Will plot all sites
import ausLib
import xarray
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
calib = 'melbourne'
total_rain = dict()
site_info = dict()
my_logger = ausLib.setup_log(1)
for site in ausLib.site_numbers.keys():
    direct = ausLib.data_dir / f'summary/{site}_rain_{calib}'
    files = sorted(direct.glob('*rain*.nc'))
    # check files exist
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f'No file {f}')

    ds = xarray.open_mfdataset(files,combine='nested',concat_dim='time')
    mean_rain = ds.mean_raw_rain_rate.mean(['x','y']).load()
    total_rain[site] = mean_rain*mean_rain.time.dt.days_in_month*24
    site_info[site] = ausLib.site_info(ausLib.site_numbers[site])
    my_logger.info(f'Loaded data for {site}')
my_logger.info('Loaded all data')




## now to plot data
fig,axs = ausLib.std_fig_axs('monthly_mean_rain',sharex=True)
for site,ax in axs.items():
    total_rain[site].plot(ax=ax,drawstyle='steps-mid')
    ax.set_ylabel('Total Rainfall (mm)')
    ausLib.plot_radar_change(ax,site_info[site])
    ax.set_title(site)

fig.show()


