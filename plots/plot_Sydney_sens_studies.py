# plot sensitivity studies for Sydney.
# Calibration -- melbourne and brisabne + melbourne calibration with larger dbz range.

import ausLib
import xarray
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import commonLib
import numpy as np
from R_python import gev_r
from plot_gev_fits import  get_data # use get_data from plot_gev_fits
from process_submit.process_gev_fits import comp_radar_fit

def proc_events(file, threshold=0.5):
    drop_vars = ['xpos', 'ypos', 'fraction', 'sample_resolution', 'height', 'Observed_temperature', 'time']
    radar_dataset = xarray.load_dataset(file, drop_variables=drop_vars)
    radar_dataset = radar_dataset.sel(resample_prd=['30min', '1h', '2h', '4h'])
    # convert radar_dataset to accumulations.

    msk = (radar_dataset.max_value > threshold)
    radar_msk = radar_dataset.where(msk)
    mn_temp = radar_msk.ObsT.mean(['quantv', 'EventTime'])
    radar_msk['Tanom'] = radar_msk.ObsT - mn_temp  #
    return radar_msk

# code now!
my_logger = ausLib.setup_log(1)
if not ('loaded_sydney_sens_studies' in locals()):
    my_logger.info('Loading data.')
    # load in the data

    gev_dirs = list((ausLib.data_dir / 'processed').glob('Sydney_rain*/fits'))
    gev_dirs = [f for f in gev_dirs if not ('check' in str(f) or 'old' in str(f))]
    gev=dict()


    for dirpath in gev_dirs:
        name = dirpath.parent.name.replace('Sydney_rain_', '').capitalize()
        gev[name] = get_data(dirpath) # using code from plot_gev_fits
        my_logger.info(f'Loaded {name}')

    # add in some sens studies on the Melbourne calibration
    my_logger.info('Doing Melbourne Sensitivity Studies')
    radar_msk = proc_events(ausLib.data_dir / 'processed' / 'Sydney_rain_melbourne' /
                            'events_seas_mean_Sydney_rain_melbourne_DJF.nc'
                            )

    # drop post 2020 data
    msk = (radar_msk.t.dt.year < 2020)

    be_t, fit_t_bs = comp_radar_fit(radar_msk.where(msk), cov=['Tanom'])
    be_t = be_t.Parameters.mean('sample')
    be_ratio = ausLib.comp_ratios(be_t)
    param_names = [k.replace('_Tanom', '') for k in be_ratio.parameter.values]
    be_ratio = be_ratio.assign_coords(parameter=param_names).rename(dict(parameter="parameter_change"))
    be_t = be_t.sel(parameter=['location', 'scale', 'shape'])
    gev['Melbourne_2020-'] = xarray.Dataset(dict(be_ratio=be_ratio,be_t=be_t))


    # and only post 2010
    msk = (radar_msk.t.dt.year >= 2010)
    be_t, fit_t_bs = comp_radar_fit(radar_msk.where(msk), cov=['Tanom'])
    be_t = be_t.Parameters.mean('sample')
    be_ratio = ausLib.comp_ratios(be_t)
    param_names = [k.replace('_Tanom', '') for k in be_ratio.parameter.values]
    be_ratio = be_ratio.assign_coords(parameter=param_names).rename(dict(parameter="parameter_change"))
    be_t = be_t.sel(parameter=['location', 'scale', 'shape'])
    gev['Melbourne_2010+'] = xarray.Dataset(dict(be_ratio=be_ratio,be_t=be_t))

    gev['Melbourne_150km'] = gev.pop('Melbourne_75km')  # rename the 75km region to 150km -- as that as what it is.
    loaded_sydney_sens_studies = True
    my_logger.info(f'Loaded all data -- memory use {ausLib.memory_use()}')


colors={
    'Melbourne_150km':'red',
    'Brisbane':'blue',
    'Melbourne_10min':'orange',
    'Melbourne_5_65':'green',
    'Melbourne_2020-':'purple',
    'Melbourne_2010+':'royalblue',
#    'Melbourne':"black"
}



## now to plot the gev ratios.
fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(7,4),num='sydney_sens_studies',clear=True,layout='constrained')
label=commonLib.plotLabel()
for ax, param in zip(axs, ['Dlocation', 'Dscale']):
    uncert = gev['Melbourne'].bs_ratio.sel(parameter_change=param).std('bootstrap_sample') * \
             scipy.stats.norm().isf(0.05)*100
    ax.fill_between(uncert.resample_prd, -uncert, uncert, color='grey', alpha=0.5)
    label.plot(ax)
    for name,color in colors.items():
        ds = gev[name]
        ratio  = ds.be_ratio.sel(parameter_change=param)
        ratio = ratio -  gev['Melbourne'].be_ratio.sel(parameter_change=param)
        ratio = ratio * 100
        ratio.plot(ax=ax, label=name, color=colors[name], marker='o',ms=6)

    ax.set_title(param)
    ax.set_xlabel('Resample period')
    ax.set_ylabel('%/K Change from reference')
    ax.axhline(0,linestyle='--',color='black')

#fig.suptitle('Sydney Sensitivity Studies')
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3,fontsize='small',loc='outside upper center',handletextpad=0.2,handlelength=1.,columnspacing=1.0)


fig.show()
commonLib.saveFig(fig)
