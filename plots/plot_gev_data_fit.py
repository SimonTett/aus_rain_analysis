# code to do prob plot for GEV fit.
import ausLib
import xarray
import commonLib
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# get in the data.
norm_data=dict()
norm_crit=dict()
shape=dict()
delta_aic=dict()
my_logger = ausLib.setup_log(1)
use_cache=False
if not (use_cache and ('loaded_gev_data_fit' in locals())):
    for site in ausLib.site_numbers.keys():
        name = site + '_rain_melbourne'
        dir = ausLib.data_dir / 'processed' / name
        events = xarray.load_dataset(dir / f'events_seas_mean_{name}_DJF.nc')  # events
        fits_t = xarray.load_dataset(dir / 'fits' / 'gev_fit_temp.nc')  # fits to data
        fits = xarray.load_dataset(dir / 'fits' / 'gev_fit.nc')  # fits to data
        delta_aic[site] = (fits_t.AIC - fits.AIC).mean('sample').drop_sel(resample_prd='8h')
        params = fits_t.Parameters.mean('sample')
        # remove small values
        L = events.max_value > 0.5
        events = events.where(L)
        # normalise the data
        anomT = events.ObsT - events.ObsT.mean('EventTime')
        loc = params.sel(parameter='location') + anomT * params.sel(parameter='Dlocation_Tanom')  # location param
        scale = params.sel(parameter='scale') + anomT * params.sel(parameter='Dscale_Tanom')  # scale param
        norm_data[site] = (events.max_value - loc) / scale
        shape[site] = params.sel(parameter='shape')

        my_logger.debug(f'Loaded {site}')
    my_logger.info('Loaded data')
    loaded_gev_data_fit = True
else:
    my_logger.info('Using cached data')

## plot the data
fig,axs = ausLib.std_fig_axs('gev_data_fit',sharex=True,sharey=True)

for site,ax in axs.items():
    for prd,col in zip(['30min','1h','2h','4h'],['r','g','b','brown']):
        data = norm_data[site].selFalse(resample_prd=prd).isel(quantv=slice(1,-1)).stack(idx=['quantv','EventTime']).dropna('idx')
        gev= scipy.stats.genextreme( c=shape[site].sel(resample_prd=prd))
        osm, osr = scipy.stats.probplot(data,dist=gev,fit=False)
        ax.scatter(osm,osr,marker='.',s=1,color=col,label=prd)
        med = gev.median()
        ax.plot(med,data.median(), color=col, marker='+', markersize=15)
    ax.axline((1,1),(10,10),color='k')
    ax.set_xlabel('Theoretical ')
    ax.set_ylabel('Empirical')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.1,100)
    ax.set_xlim(0.1,100)
    ax.label_outer()
    ax.set_title(site)

    # add an inset plot for the AIC changes
    ax_inset = ax.inset_axes([0.125,0.65,0.35,0.3],zorder=0)
    (-delta_aic[site]/1000).plot(ax=ax_inset,color='k',marker='o')
    print(site,delta_aic[site].values)
    ax_inset.set_title(r'$-\Delta$ AIC $\times 10^{-3}$',size='small',pad=0)
    ax_inset.set_ylabel('')
    ax_inset.set_xlabel('')
    #ax_inset.set_ylim(-50,0)
    ax_inset.tick_params(axis='x',labelsize='x-small',labelrotation=45)
    ax_inset.tick_params(axis='y',labelsize='x-small',pad=0,direction='in')
    ax_inset.axhline(0,color='k',linestyle='--')
    ax_inset.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_inset.label_outer()
    # done making indivisual plots.
#general figure stuff
handles, labels = axs['Melbourne'].get_legend_handles_labels()
for handle in handles:
    handle._sizes= [10.0] # ass suggested by chatGPT
legend=fig.legend(handles, labels, loc=(0.4, 0.9), fontsize='small')

fig.show()
commonLib.saveFig(fig)