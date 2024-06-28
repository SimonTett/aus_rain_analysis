# plot the GEV fits for the event extremes.
# will focus on the fractional changes and try to plot the 30min, 1hr & 2 hr data.
## load in the data
import matplotlib.pyplot as plt
import xarray
import ausLib
import scipy.stats
import numpy as np

import commonLib

my_logger  = ausLib.setup_log(1)

gev=dict() # Where we store the data.
best_est=dict()
bootstrap=dict()

# mornington looks weird - need to examine it more carefully. Prob look at events.



for site in ausLib.site_numbers.keys():
    name = f'{site}_rain_melbourne' # most sites calibrated using Melbourne rain
    if site in ['Melbourne','Brisbane','Wtakone']: # need to rerun these cases using Melbourne rain transfer fn
        name = f'{site}_rain'
    gev_t_file = ausLib.data_dir / f'processed/{name}/fits/gev_fit_temp.nc'
    gev_t_bs_file = ausLib.data_dir / f'processed/{name}/fits/gev_fit_temp_bs.nc'
    gev_file = ausLib.data_dir / f'processed/{name}/fits/gev_fit.nc'
    gev_bs_file = ausLib.data_dir / f'processed/{name}/fits/gev_fit_bs.nc'
    be_t = xarray.load_dataset(gev_t_file).mean(dim='sample').drop_vars('postchange_start',errors='ignore')
    best_est[site] = be_t
    bs_t = xarray.load_dataset(gev_t_bs_file).drop_vars('postchange_start',errors='ignore')
    bootstrap[site] = bs_t
    be = xarray.load_dataset(gev_file).mean(dim='sample').drop_vars('postchange_start',errors='ignore')
    bs = xarray.load_dataset(gev_bs_file).drop_vars('postchange_start',errors='ignore')



    # compute the fractional location and shape changes + uncerts.

    ds=dict()
    for p in ['location', 'scale']:
        dp = f'D{p}_Tanom'
        dfract_bs = bs_t.Parameters.sel(parameter=dp) / bs_t.Parameters.sel(parameter=p)
        ds[p+'_std'] = dfract_bs.std(dim='bootstrap_sample')
        ds[p] = be_t.Parameters.sel(parameter=dp) / be_t.Parameters.sel(parameter=p)
    ds['delta_AIC'] = (be_t.AIC - be.AIC)
    ds['delta_AIC_std'] = (bs_t.AIC - bs.AIC).std(dim='bootstrap_sample')
    ds['pImp'] =( bs_t.AIC  < bs.AIC).sum(dim='bootstrap_sample')/bs.AIC.count('bootstrap_sample')
    ds['p'] = np.exp((be_t.AIC - be.AIC)/2)
    ds=xarray.Dataset(ds)
    gev[site] = ds


    my_logger.info(f'Processed {site}')

## now to plot the data
fig,axes = ausLib.std_fig_axs(f'GEV_change',sharey=True,sharex=True)

for site,ax in axes.items():
    pbetter = gev[site].pImp.drop_sel(resample_prd='8h')
    print(site,gev[site].to_dataframe().loc[:,['p','pImp']].round(2).T)
    ms = np.where(pbetter > 0.9, 20, 0)
    for p,m,c in zip(['location', 'scale'],['o','h'],['blue','red']):
        mn = gev[site][p].drop_sel(resample_prd='8h')*100
        sd = gev[site][p+'_std'].drop_sel(resample_prd='8h')*100

        ax.errorbar(mn.resample_prd,mn,yerr=1.3*sd,label=p,linestyle='--',color=c,capsize=5,elinewidth=2)
        ax.scatter(mn.resample_prd,mn,s=ms,color=c,marker=m)
        for v in [0,7.5,15]:
            ax.axhline(v,linestyle='--',color='black')
            ax.set_ylim(-30,50)
    ax.set_ylabel('%/K')
    ax.set_xlabel('')
    ax.set_title(site)
    #print(site,gev[site].to_dataframe().loc[:, ['delta_AIC', 'delta_AIC_std']].round(-1))
axes['Melbourne'].legend()
fig.suptitle('Fractional changes in GEV location and scale parameters.')
fig.show()
commonLib.saveFig(fig)

## let's make a plot of rtn periods for all accums
import pandas as pd
resample_hours = pd.to_timedelta(be_t.resample_prd)/pd.Timedelta(1,'h')
# and convert it back to a dataarray
resample_hours = xarray.DataArray(resample_hours, coords={'resample_prd':be_t.resample_prd})
from R_python import gev_r
fig,axs = ausLib.std_fig_axs(f'GEV_rtn_value',sharex=True,sharey=True)
for site,ax in axs.items():
    gev_be = best_est[site]
    ds = gev_r.xarray_gev_isf(gev_be.Parameters, 1.0 / np.geomspace(1000,10))
    ds = ds.drop_sel(resample_prd='8h')*resample_hours # drop 8h and convert to accum over period.
    lines=ds.plot.line(x='pvalues',ax=ax,add_legend=False)
    ax.set_ylabel('DJF Max Rainfall (mm)')
    ax.set_xlim(1e-1,1e-3)
    ax.set_xlabel('P value')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(site)
    gev_bs = bootstrap[site]
fig.legend(lines,ds.resample_prd.values,ncol=2,loc=(0.4,0.9),fontsize='small')
fig.suptitle('Return values (mm/h)')
fig.show()
commonLib.saveFig(fig)
