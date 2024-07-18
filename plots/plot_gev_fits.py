# plot the GEV fits for the event extremes.
# will focus on the fractional changes and try to plot the 30min, 1hr & 2 hr data.
## load in the data
import matplotlib.pyplot as plt
import pandas as pd
import xarray
import ausLib
import scipy.stats
import numpy as np

import commonLib
import pathlib

my_logger = ausLib.setup_log(1)
def read_data(file: pathlib.Path) -> xarray.Dataset:
    """
    Read in the fit data and filter out bad fits.
    Args:
        file: File to read in generated by gev_fit

    Returns: xarray.Dataset containing the fit data after remove of cases with -ve locations

    """
    ds = xarray.load_dataset(file).reindex(resample_prd=['30min', '1h', '2h', '4h'])  # order sensible and drop 8h
    #L = ds.Parameters.sel(parameter='scale') > -10
    L = (ds.Parameters.sel(parameter='location') > 0)
    if L.sum() < L.count():
        my_logger.warning(f'Filtering out {int(L.count() - L.sum())} bad fits for {file}')
    ds = ds.where(L)  #.dropna('sample', how='all')
    return ds.mean(dim='sample').drop_vars('postchange_start', errors='ignore')


def get_data(fit_dir: pathlib.Path, boostrap: bool = True) -> xarray.Dataset:
    be_t = read_data(fit_dir / 'gev_fit_temp.nc')
    be = read_data(fit_dir / 'gev_fit.nc')

    # compute the fractional location and shape changes
    ds = dict()
    ds['be_t'] = be_t.Parameters.sel(parameter=['location', 'scale', 'shape'])
    ds['delta_AIC'] = (be_t.AIC - be.AIC)
    ds['p'] = np.exp((be_t.AIC - be.AIC) / 2)
    be_ratio = ausLib.comp_ratios(be_t.Parameters)
    param_names = [k.replace('_Tanom', '') for k in be_ratio.parameter.values]
    be_ratio = be_ratio.assign_coords(parameter=param_names).rename(dict(parameter="parameter_change"))
    ds['be_ratio'] = be_ratio
    if boostrap:  # read in the boot strap data.
        gev_bs_file = fit_dir / 'gev_fit_bs.nc'
        gev_t_bs_file = fit_dir / 'gev_fit_temp_bs.nc'
        bs = read_data(gev_bs_file)
        bs_t = read_data(gev_t_bs_file)
        ds['p_bs'] = np.exp((bs_t.AIC - bs.AIC) / 2)
        ds['bs_t'] = bs_t.Parameters.sel(parameter=['location', 'scale', 'shape'])
        ds['bs_delta_AIC'] = (bs_t.AIC - bs.AIC)
        bs_ratio = ausLib.comp_ratios(bs_t.Parameters)  #.std('bootstrap_sample')
        # now to rename
        bs_ratio = bs_ratio.assign_coords(parameter=param_names).rename(dict(parameter="parameter_change"))
        ds['bs_ratio'] = bs_ratio
        my_logger.debug(f'Loaded BS datat from {fit_dir}')
    my_logger.info(f'Loaded {fit_dir}')
    return xarray.Dataset(ds)


# Main code
if __name__ == '__main__':


    gev = dict()  # Where we store the data.

    uncert = 0.05  # 5-95%
    uncert_sd_scale = scipy.stats.norm().isf(uncert)  # roughly 5-95% range
    quants = [uncert, 0.5, 1 - uncert]
    # mornington looks weird – need to examine it more carefully. Prob look at events. Some oddities in the fit.
    # so filtering out shapes < -10. I think it is problems with the fit.
    end_name_sens = '_rain_brisbane'  # calibration
    end_name = '_rain_melbourne'

    for site in ausLib.site_numbers.keys():
        name = site + end_name
        path = ausLib.data_dir / f'processed/{name}/fits'
        gev[site] = get_data(path)
        sens_path = ausLib.data_dir / f'processed/{site}{end_name_sens}/fits'
        sens = get_data(sens_path, boostrap=False)
        rename = {k: k + '_sens' for k in sens.data_vars}
        sens = sens.rename(rename)
        gev[site] = xarray.merge([gev[site], sens])

        my_logger.info(f'Processed {site}')

    ## now to generate the regional changes.

    for reg_name, sites in ausLib.region_names.items():
        # first extract the datasets for the sites wanted
        site_ds = []
        for site in sites:
            if site not in gev:
                my_logger.warning(f'No data for {site}')
                continue
            site_ds += [gev[site].expand_dims(site=[site])]
        if len(site_ds) == 0:
            my_logger.warning(f'No data for {reg_name}')
            continue
        site_ds = xarray.concat(site_ds, dim='site').mean(dim='site')
        gev[reg_name] = xarray.Dataset(site_ds)
    my_logger.info('Processed regions')
    ## now to plot the data

    fig, axes = ausLib.std_fig_axs(f'GEV_change', regions=True, sharey=True, sharex=True)

    for site, ax in axes.items():
        if site not in gev:
            my_logger.warning(f'No data for {site}')
            continue
        try:
            ms = np.where(gev[site].delta_AIC < 100, 20, 5)
        except AttributeError:
            ms = 20

        z_0 = dict()
        z_cc = dict()
        z_sup_cc = dict()
        for p, m, c in zip(['Dlocation', 'Dscale'], ['o', 'h'], ['blue', 'red']):
            mn = gev[site].be_ratio.sel(parameter_change=p) * 100
            mn_sens = gev[site].be_ratio_sens.sel(parameter_change=p) * 100
            sd = gev[site].bs_ratio.sel(parameter_change=p).std('bootstrap_sample') * 100
            ax.errorbar(mn.resample_prd, mn, yerr=uncert_sd_scale * sd, label=p, linestyle='--', color=c, capsize=5,
                        elinewidth=2
                        )
            ax.scatter(mn_sens.resample_prd, mn_sens, s=ms * 4, color=c, marker='*')
            scale_sd = uncert_sd_scale * sd
            z_0[p] = mn / scale_sd
            z_cc[p] = (mn - 7.5) / scale_sd
            z_sup_cc[p] = (mn - 15) / scale_sd
        # add on the CC lines.
        for v in [0, 7.5, 15]:
            ax.axhline(v, linestyle='--', color='black')
            ax.set_ylim(-30, 50)
        ax.set_ylabel('%/K')
        ax.set_xlabel('Accum Prd')
        # work out colour for site
        color = 'black'
        rp = '1h'
        if z_0['Dscale'].sel(resample_prd=rp) > 1.:
            color = 'green'
        if z_cc['Dscale'].sel(resample_prd=rp) > 1.:
            color = 'orange'
        if z_sup_cc['Dscale'].sel(resample_prd=rp) > 1.:
            color = 'red'

        ax.set_title(site, weight='bold', color=color)
        ax.label_outer()

    handles, labels = axes['Melbourne'].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.4, 0.9), fontsize='small')
    fig.suptitle('Fractional changes in GEV location and scale parameters.')
    fig.show()
    commonLib.saveFig(fig)
    commonLib.saveFig(fig,transpose=True)

    ## plot the parameters for the radar stations.
    ds = gev['Melbourne'].be_t
    resample_hours = pd.to_timedelta(ds.resample_prd) / pd.Timedelta(1, 'h')
    # and convert it back to a dataarray
    resample_hours = xarray.DataArray(resample_hours, coords={'resample_prd': ds.resample_prd})
    fig, axs = ausLib.std_fig_axs(f'GEV_params', sharex=True, sharey=True)
    for site, ax in axs.items():
        params = gev[site].be_t
        params_sens = gev[site].be_t_sens
        params_bs = gev[site].bs_t
        for p, m, c in zip(['location', 'scale', 'shape'], ['o', 'h', '*'], ['blue', 'red', 'purple']):
            mn = params.sel(parameter=p)
            mn_sens = params_sens.sel(parameter=p)
            q_bs = params_bs.sel(parameter=p).quantile(quants, 'bootstrap_sample')
            if p == 'shape':
                mn = -mn
                mn_sens = -mn_sens
                q_bs = -q_bs.reindex(quantile=list(reversed(q_bs['quantile'])))  # get into Cole's convention.
                a = ax.twinx()
                a.set_ylabel('Shape', color=c, size='small')
                a.set_ylim(-0.8, 0.8)
                a.tick_params(axis='y', labelcolor=c, labelsize='small')
                a.axhline(0.0, linestyle='dotted', color=c)
                a.label_outer()
            else:
                a = ax
                a.set_ylabel('Loc/Scale (mm)', size='small')
                mn = mn * resample_hours
                mn_sens = mn_sens * resample_hours
                a.set_ylim(0, 20.)
                a.tick_params(axis='y', labelcolor='k', labelsize='small')

            err = np.abs(q_bs.isel(quantile=[0, -1]) - q_bs.sel(quantile=0.5))
            a.errorbar(mn.resample_prd, mn, yerr=err,
                       linestyle='--', color=c, capsize=5, elinewidth=2, label=p
                       )
            a.scatter(mn_sens.resample_prd, mn_sens, s=30, color=c, marker='*')
            a.set_xlabel('Accum Prd')
            a.label_outer()

        ax.set_title(site)

    fig.suptitle('GEV Parameters')
    fig.show()
    commonLib.saveFig(fig)
    commonLib.saveFig(fig,transpose=True)
