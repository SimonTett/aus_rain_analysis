# plot the GEV fits for the event extremes.
# will focus on the fractional changes and try to plot the 30min, 1hr & 2 hr data.
## load in the data
import matplotlib.pyplot as plt
import xarray
import ausLib
import scipy.stats
import numpy as np

import commonLib
import pathlib

my_logger = ausLib.setup_log(1)

gev = dict()  # Where we store the data.
best_est = dict()
best_est_sens=dict()
bootstrap = dict()
uncert = 0.05  # 5-95%
uncert_sd_scale = scipy.stats.norm().isf(uncert)  # roughly 5-95% range
quants = [uncert, 0.5, 1 - uncert]
# mornington looks weird – need to examine it more carefully. Prob look at events. Some oddities in the fit.
# so filtering out shapes < -10. I think it is problems with the fit.
end_name_sens = '_rain_brisbane'  # calibration
end_name = '_rain_melbourne'


def trim(ds: xarray.Dataset) -> xarray.DataArray:
    return ds.Parameters.where(ds.Parameters.sel(parameter='scale') > -10)


def read_data(file: pathlib.Path) -> xarray.DataArray:
    ds = xarray.load_dataset(file).reindex(resample_prd=['30min','1h','2h','4h','8h'])
    L = ds.Parameters.sel(parameter='scale') > -10
    L = L & (ds.Parameters.sel(parameter='location') > 0)
    if L.sum() < L.count():
        my_logger.warning(f'Filtering out {int(L.count() - L.sum())} bad fits for {file}')
    ds = ds.where(L)#.dropna('sample', how='all')
    return ds.mean(dim='sample').drop_vars('postchange_start', errors='ignore')


for site in ausLib.site_numbers.keys():
    name = site + end_name
    gev_t_file = ausLib.data_dir / f'processed/{name}/fits/gev_fit_temp.nc'
    gev_t_bs_file = ausLib.data_dir / f'processed/{name}/fits/gev_fit_temp_bs.nc'
    gev_file = ausLib.data_dir / f'processed/{name}/fits/gev_fit.nc'
    gev_bs_file = ausLib.data_dir / f'processed/{name}/fits/gev_fit_bs.nc'
    be_t = read_data(gev_t_file)
    best_est[site] = be_t
    bs_t = read_data(gev_t_bs_file)
    bootstrap[site] = bs_t
    be = read_data(gev_file)
    bs = read_data(gev_bs_file)
    # get in the alternative calibration -- no bootstrap just the mean.
    name = site + end_name_sens
    gev_t_file = ausLib.data_dir / f'processed/{name}/fits/gev_fit_temp.nc'
    gev_file = ausLib.data_dir / f'processed/{name}/fits/gev_fit.nc'
    be_t_sens = read_data(gev_t_file)
    be_sens = read_data(gev_file)
    best_est_sens[site] = be_t_sens

    # compute the fractional location and shape changes + uncerts.

    ds = dict()

    ds['delta_AIC'] = (be_t.AIC - be.AIC)
    ds['delta_AIC_std'] = (bs_t.AIC - bs.AIC).std(dim='bootstrap_sample')
    ds['pImp'] = (bs_t.AIC < bs.AIC).sum(dim='bootstrap_sample') / bs.AIC.count('bootstrap_sample')
    ds['p'] = np.exp((be_t.AIC - be.AIC) / 2)
    ds = xarray.Dataset(ds)
    be_ratio = ausLib.comp_ratios(be_t.Parameters)
    be_ratio = be_ratio.assign_coords(parameter=[k.replace('_Tanom','') for k in be_ratio.parameter.values]).rename('be_ratio')
    ds['be_ratio'] = be_ratio
    be_ratio_sens=ausLib.comp_ratios(be_t_sens.Parameters)
    be_ratio_sens = be_ratio_sens.assign_coords(parameter=[k.replace('_Tanom','') for k in be_ratio_sens.parameter.values]).rename('be_ratio_sens')
    ds['be_ratio_sens'] = be_ratio_sens
    bs_ratio = ausLib.comp_ratios(bs_t.Parameters)#.std('bootstrap_sample')
    # now to rename
    bs_ratio = bs_ratio.assign_coords(parameter=[k.replace('_Tanom','') for k in bs_ratio.parameter.values]).rename('bs_uncert')
    ds['bs_ratio']=bs_ratio
    ds['bs_ratio_std'] = bs_ratio.std('bootstrap_sample')
    gev[site] = ds

    my_logger.info(f'Processed {site}')
## now to generate the regional changes.


for rng, sites in ausLib.region_names.items():
    # first extract the datasets for the sites wanted
    site_ds = []
    for site in sites:
        if site not in gev:
            my_logger.warning(f'No data for {site}')
            continue
        site_ds += [gev[site].expand_dims(site=[site])]
    site_ds = xarray.concat(site_ds, dim='site')
    bs_mean = site_ds.bs_ratio.mean(dim='site')
    be_mean_sens = site_ds.be_ratio_sens.mean(dim='site')
    ds = dict(be_ratio= site_ds.be_ratio.mean(dim='site'), be_ratio_sens=be_mean_sens,
              bs_ratio=bs_mean,bs_ratio_std=bs_mean.std('bootstrap_sample'))


    gev[rng] = xarray.Dataset(ds)
my_logger.info('Processed regions')

## now to plot the data

fig, axes = ausLib.std_fig_axs(f'GEV_change', regions=True, sharey=True, sharex=True)

for site, ax in axes.items():
    if site not in gev:
        my_logger.warning(f'No data for {site}')
        continue
    try:
        pbetter = gev[site].pImp.drop_sel(resample_prd='8h')
        print(site, gev[site].to_dataframe().loc[:, ['p', 'pImp']].round(2).T)
        ms = np.where(pbetter > 0.9, 20, 0)
    except AttributeError:
        ms = 20
    for p, m, c in zip(['Dlocation', 'Dscale'], ['o', 'h'], ['blue', 'red']):
        mn = gev[site].be_ratio.sel(parameter=p).drop_sel(resample_prd='8h') * 100
        mn_sens = gev[site].be_ratio_sens.sel(parameter=p).drop_sel(resample_prd='8h') * 100
        sd = gev[site].bs_ratio_std.sel(parameter=p).drop_sel(resample_prd='8h') * 100

        ax.errorbar(mn.resample_prd, mn, yerr=uncert_sd_scale * sd, label=p, linestyle='--', color=c, capsize=5, elinewidth=2)
        ax.scatter(mn.resample_prd, mn, s=ms, color=c, marker=m)
        ax.scatter(mn_sens.resample_prd, mn_sens, s=ms*2, color=c, marker='*')
        # add on the
        for v in [0, 7.5, 15]:
            ax.axhline(v, linestyle='--', color='black')
            ax.set_ylim(-30, 50)
    ax.set_ylabel('%/K')
    ax.set_xlabel('')
    ax.set_title(site)
    #print(site,gev[site].to_dataframe().loc[:, ['delta_AIC', 'delta_AIC_std']].round(-1))
handles, labels = axes['Melbourne'].get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.4, 0.9), fontsize='small')
fig.suptitle('Fractional changes in GEV location and scale parameters.')
fig.show()
commonLib.saveFig(fig)

## let's make a plot of rtn periods for all accums
import pandas as pd

resample_hours = pd.to_timedelta(be_t.resample_prd) / pd.Timedelta(1, 'h')
# and convert it back to a dataarray
resample_hours = xarray.DataArray(resample_hours, coords={'resample_prd': be_t.resample_prd})
colors = ['blue', 'orange', 'green', 'red']
from R_python import gev_r

fig, axs = ausLib.std_fig_axs(f'GEV_rtn_value', sharex=True, sharey=True)
pv = 1.0 / np.geomspace(100, 5)
for site, ax in axs.items():
    gev_be = best_est[site]
    ds = gev_r.xarray_gev_isf(gev_be.Parameters, pv).drop_sel(resample_prd='8h') * resample_hours  # drop 8h and convert to accum over period.
    gev_be_sens = best_est_sens[site]
    ds_sens = gev_r.xarray_gev_isf(gev_be_sens.Parameters, pv).drop_sel(resample_prd='8h') * resample_hours
    # drop 8h and convert to accum over period.

    #lines = ds.plot.line(x='pvalues', ax=ax, add_legend=False,colors=colors)

    gev_bs = gev_r.xarray_gev_isf(bootstrap[site].Parameters, pv) * resample_hours
    #sd = gev_bs.std(dim='bootstrap_sample').drop_sel(resample_prd='8h')
    quant = gev_bs.quantile(quants, dim='bootstrap_sample').drop_sel(resample_prd='8h')
    #quant = quant+ds-quant.sel(quantile=0.5) # bias correct bootstrap.
    #lower,upper = (ds - uncert_sd * sd, ds + uncert_sd * sd)
    lower, upper = (quant.isel(quantile=0), quant.isel(quantile=-1))
    lines = []
    for resample, col in zip(sd.resample_prd, colors):
        #for resample,col in zip(['1h'],colors):
        uncert_sd = sd.sel(resample_prd=resample)
        ax.fill_between(ds.pvalues, lower.sel(resample_prd=resample), upper.sel(resample_prd=resample), alpha=0.3,
                        color=col
                        )
        lines.extend(
            ds.sel(resample_prd=resample).plot.line(x='pvalues', ax=ax, add_legend=False, color=col, linewidth=2,
                                                    label=resample
                                                    )
            )
        ds_sens.sel(resample_prd=resample).plot.line(x='pvalues', ax=ax, add_legend=False, color=col, linestyle='--',
                                                     linewidth=2,marker='x',markevery=10,
                                                     )
    ax.set_ylabel('DJF Max Rainfall (mm)')
    ax.set_xlim(1/5, 1.2e-2)
    ax.set_ylim(1, 100.)
    ax.set_xlabel('P value')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(site)

fig.legend(lines, sd.resample_prd.values, ncol=2, loc=(0.4, 0.9), fontsize='small')
fig.suptitle('Return values (mm/h)')
fig.show()
commonLib.saveFig(fig)

## plot the parameters for the regions.
fig, axs = ausLib.std_fig_axs(f'GEV_params', sharex=True, sharey=True)
for site, ax in axs.items():
    params = best_est[site].Parameters.drop_sel(resample_prd='8h')
    params_sens = best_est_sens[site].Parameters.drop_sel(resample_prd='8h')
    params_bs = bootstrap[site].Parameters.drop_sel(resample_prd='8h')
    uncert_sd = params_bs.std(dim='bootstrap_sample')
    for p, m, c in zip(['location', 'scale','shape'], ['o', 'h','*'], ['blue', 'red','orange']):
        mn = params.sel(parameter=p)
        mn_sens = params_sens.sel(parameter=p)
        q_bs = params_bs.sel(parameter=p).quantile(quants,'bootstrap_sample')
        if p == 'shape':
            mn = -mn
            mn_sens = -mn_sens
            q_bs = -q_bs.reindex(quantile=list(reversed(q_bs['quantile'])) ) # get into Cole's convention.
            a=ax.twinx()
            a.set_ylabel('Shape',color=c)
            a.set_ylim(-0.8,0.8)
            a.axhline(0.0,linestyle='dotted',color=c)
        else:
            a=ax
            a.set_ylabel('Loc/Scale (mm)')
            mn = mn*resample_hours
            mn_sens = mn_sens*resample_hours
            sd = sd*resample_hours
            a.set_ylim(0,20.)
        err = np.abs(q_bs.isel(quantile=[0,-1])-q_bs.sel(quantile=0.5))
        a.errorbar(mn.resample_prd, mn, yerr=err,
                    linestyle='--', color=c, capsize=5, elinewidth=2,label=p )
        a.scatter(mn_sens.resample_prd, mn_sens, s=30, color=c, marker='*')
        a.set_xlabel('Accum Prd')
        a.label_outer()


    ax.set_title(site)


fig.suptitle('Gev Params')
fig.show()
commonLib.saveFig(fig)


