# plot the AIC changes from various covariates over the no-covariate option.
# Will plot best two models for each covariate.
import ausLib
import matplotlib.pyplot as plt
ausLib.setup_log(2) # have debug logging
endfiles={k:'_'+v+'.nc'for k,v in ausLib.non_default_dates.items()}

gev_file = lambda site: ausLib.data_dir / 'processed' / site/'gev_fits' / \
                        f'gev_fit{endfiles.get(site.split('_')[0],".nc")}'
fit = ausLib.read_process(seas_file_fn=gev_file)
fit_cov= dict()
delta_aic = dict()
for cov in ['temperature','dewpoint','sample_resolution','fraction']:
    gev_cov_file = lambda site: ausLib.data_dir / 'processed' / site/'gev_fits' / \
                                f'gev_fit_{cov}{endfiles.get(site.split('_')[0],'.nc')}'
    fit_cov[cov] = ausLib.read_process(seas_file_fn=gev_cov_file )
    delta_aic[cov]={k:(v.AIC/fit[k].AIC).mean('sample')-1 for k,v in fit_cov[cov].items()}

## plot the AIC changes
fig,axs = ausLib.std_fig_axs('gev_aic_change',sharex=True,sharey=True)
for site,ax in axs.items():
    name = site + '_rain_melbourne'
    for cov,color in zip(['temperature','dewpoint','sample_resolution','fraction'],['red','purple','black','green']):
        ax.plot(100*delta_aic[cov][name],color=color,marker='x',ms=10,label=cov)
    ax.set_title(site)
    ax.set_xlabel('resample')
    ax.set_ylabel('AIC change (%)')
    ax.label_outer()
    ax.set_ylim(-20,5)
    #ax.set_yscale('log')
# Grab the labels and add them to the figure legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
fig.show()




