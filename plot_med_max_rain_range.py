# plot all stations  pre and post most recent radar change as fn of range

import matplotlib.pyplot as plt
import ausLib
import xarray
import numpy as np
import commonLib
import pathlib

mx_data = dict()
mean_rain = dict()
conversion = '_rain_melbourne'
long_radar_data = ausLib.read_radar_file("meta_data/long_radar_stns.csv")
sites = dict()
for site, id in ausLib.site_numbers.items():
    site_info = ausLib.site_info(id).iloc[-1]
    yr = site_info.postchange_start.year
    sites[site] = yr - 1

for site in sites.keys():
    name = site + conversion
    seas_file = ausLib.data_dir / f'processed/{name}/seas_mean_{name}_DJF.nc'
    if not seas_file.exists():
        raise FileNotFoundError(f'No season  file for {site} with {seas_file}')
    mx_rain = xarray.open_dataset(seas_file).max_rain_rate.load()
    # mask out small extremes
    mx_rain = mx_rain.where(mx_rain > 1)
    mn_rain = xarray.open_dataset(seas_file).mean_raw_rain_rate.load() * 91 * 24  # convert to mm
    mn_rain.attrs['units'] = 'mm'
    r = np.sqrt(mx_rain.x ** 2 + mx_rain.y ** 2) / 1000.
    mx_rain['range'] = r
    mn_rain['range'] = r
    mx_data[site] = mx_rain.groupby_bins('range', np.arange(0, 150, 10)).median().load()
    mean_rain[site] = mn_rain.groupby_bins('range', np.arange(0, 150, 10)).mean().load()

fig, axs = ausLib.std_fig_axs('med_max_rain_range')
#plt.subplots(2,2,figsize=(8,6),sharex=True,sharey=True,clear=True,layout='constrained',num='med_max_rain_range')
for (site, break_yr) in sites.items():
    ax = axs[site]
    pre = mx_data[site].sel(resample_prd='4h', time=slice(None, f'{break_yr}'))
    post = mx_data[site].sel(resample_prd='4h', time=slice(f'{break_yr + 1}', None))
    if len(pre.time) > 0:
        #pre.plot.line(x='range_bins', ax=ax, linestyle='-', add_legend=False, label='_None')
        pre.median('time').plot(x='range_bins', ax=ax, linestyle='-', label=f"Pre {break_yr}", color='k', linewidth=3)
    if len(post.time) > 0:
        #post.plot.line(x='range_bins', ax=ax, linestyle='--', label="_None", add_legend=False)
        post.median('time').plot(x='range_bins', ax=ax, linestyle='--', label=f"Post {break_yr}", color='k',
                                 linewidth=3
                                 )
    ax.set_title(site)
    #ax.set_yscale('log')
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Rainfall (mm/h)')
    ax.set_ylim(0,20)
    ax.legend()
fig.suptitle('Rx1H')
fig.show()
commonLib.saveFig(fig, savedir=pathlib.Path('extra_figs'))
commonLib.saveFig(fig, savedir=pathlib.Path('extra_figs'), transpose=True)

fig, axs = ausLib.std_fig_axs('mean_rain_range', clear=True, sharex=True, sharey=True)

for (site, break_yr) in sites.items():
    ax = axs[site]
    pre = mean_rain[site].sel(time=slice(None, f'{break_yr}'))
    post = mean_rain[site].sel(time=slice(f'{break_yr + 1}', None))
    pre_yrs = len(pre.time)
    post_yrs = len(post.time)
    if pre_yrs > 0:
        #pre.plot.line(x='range_bins',ax=ax,linestyle='-',add_legend=False,label='_None',alpha=0.5)

        sd = pre.std('time') / np.sqrt(pre_yrs)
        mn = pre.mean('time')

        x = [v.mid for v in pre.range_bins.values]
        ax.fill_between(x, mn - 2 * sd, mn + 2 * sd, alpha=0.5, color='red')
        mn.plot(x='range_bins', ax=ax, linestyle='-', label=f"Pre {break_yr} N={pre_yrs}", color='red', linewidth=3)
    if post_yrs > 0:
        #post.plot.line(x='range_bins',ax=ax,linestyle='--',label="_None",add_legend=False,alpha=0.5)

        sd = post.std('time') / np.sqrt(post_yrs)
        mn = post.mean('time')

        x = [v.mid for v in post.range_bins.values]
        ax.fill_between(x, (mn - 2 * sd).values, (mn + 2 * sd).values, alpha=0.5, color='k')
        mn.plot(x='range_bins', ax=ax, linestyle='--', label=f"Post {break_yr} N={post_yrs}", color='k', linewidth=3)
    ax.set_title(site)
    #ax.set_yscale('log')
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Rainfall (mm)')

    ax.legend()
fig.suptitle('Total DJF rain')
fig.show()
commonLib.saveFig(fig, savedir=pathlib.Path('extra_figs'))
commonLib.saveFig(fig, savedir=pathlib.Path('extra_figs'), transpose=True)

# lets do all the Sydney sens studies.
files = list((ausLib.data_dir / 'processed').glob('Sydney_*/seas_mean_Sydney*_DJF.nc'))
fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True, clear=True, num='Sydney_sens_mean_range')
break_year = sites['Sydney']
for file, ax in zip(files, axs.flatten()):

    mn_rain = xarray.open_dataset(file).mean_raw_rain_rate.load() * 91 * 24  # convert to mm
    mn_rain.attrs['units'] = 'mm'
    r = np.sqrt(mn_rain.x ** 2 + mn_rain.y ** 2) / 1000.
    mn_rain['range'] = r
    mn_rain = mn_rain.groupby_bins('range', np.arange(0, 150, 10)).mean().load()
    pre = mn_rain.sel(time=slice(None, f'{break_year}'))
    post = mn_rain.sel(time=slice(f'{break_year + 1}', None))
    pre_yrs = len(pre.time)
    post_yrs = len(post.time)
    if pre_yrs > 0:
        sd = pre.std('time') / np.sqrt(pre_yrs)
        mn = pre.mean('time')
        x = [v.mid for v in pre.range_bins.values]
        mn.plot(x='range_bins', ax=ax, linestyle='-', label=f"Pre {break_year} N={pre_yrs}", color='red', linewidth=3)
        ax.fill_between(x, mn - 2 * sd, mn + 2 * sd, alpha=0.5, color='red')

    if post_yrs > 0:
        sd = post.std('time') / np.sqrt(post_yrs)
        mn = post.mean('time')
        mn.plot(x='range_bins', ax=ax, linestyle='--', label=f"Post {break_year} N={post_yrs}", color='k', linewidth=3)
        x = [v.mid for v in post.range_bins.values]
        ax.fill_between(x, (mn - 2 * sd).values, (mn + 2 * sd).values, alpha=0.5, color='k')

    ax.set_title(file.parent.name)
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Rainfall (mm)')
    ax.legend()
    ax.set_ylim(0, 300)
fig.suptitle('Sydney sens studies: Total DJF rain')

commonLib.saveFig(fig, savedir=pathlib.Path('extra_figs'))
