# Do KDE smoothing of events.
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import xarray

import ausLib

import seaborn as sns
from matplotlib.ticker import MaxNLocator

import commonLib

import cftime
my_logger = ausLib.my_logger

commonLib.init_log(my_logger,level='DEBUG')
site = 'Melbourne'
radar_dataset = xarray.load_dataset(ausLib.data_dir/f"events/{site}_hist_gndrefl_DJF.nc") # load the processed radar
radar_dataset = radar_dataset.where(radar_dataset.t.dt.year > 2000) # drop the first years.
topog = xarray.load_dataset(ausLib.data_dir/f'ancil/{site}_cbb_dem.nc').elevation
topog = topog.coarsen(x=4, y=4, boundary='trim').mean()
my_logger.debug(f"Loaded datasets")
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=[8, 8],clear=True,
                        num='kde_smooth_events',sharex='col', sharey='col',layout='constrained')

fig.get_layout_engine().set(rect=[0.05,0.0,0.95,1.0])#.execute(fig)

labels = commonLib.plotLabel()
plot_col_titles=True
for q,axis in zip([0.1,0.5,0.9],axs):
    sel = dict(quantv=q,method='nearest') # what we want!
    radar_quant = radar_dataset.sel(**sel)
    radar_quant['Area'] = radar_quant.count_cells*16.0
    radar_quant['Solar Hour'] = ((radar_quant.t.dt.hour-10)+48)%24 # convert to solar hour.
    my_logger.debug(f"Computed area, hour")
    pos = axis[0].get_position()
    y=(pos.ymax+pos.ymin)/2
    x=0.02
    fig.text(x,y,f'Quant {q}',ha='left',va='center',rotation=90,fontsize=10)
    colors=['brown','firebrick','red','darkorange','orange']
    for ax,var,xlabel,kdeplot_args in zip(axis.flatten(),
                        ['Solar Hour','Area','height','max_value'],
                        ['Solar Hour','Area (km$^2$)','Height (m)','Mx Ref (mm^6/m^3)'],
                        [ dict(gridsize=24,clip=(0,24)), # hours
                          dict(gridsize=50,log_scale=(10,None)), # area
                          dict(gridsize=50,clip=(0,1000),log_scale=(10,None)), # ht
                          dict(gridsize=50,clip=(20,None),log_scale=(10,10)) # refl
                          ]):

        for prd,color in zip(radar_quant.resample_prd.values[0:-1],colors): # don;t want the last resample prd.
            if var == 'Area':
                da = radar_quant[var].dropna('EventTime')
            else:
                da = radar_quant[var].sel(resample_prd=prd).dropna('EventTime')
            kdeplot_args.update(color=color,linewidth=1,cut=0,label=prd)
            sns.kdeplot(da.values.flatten(),ax=ax,common_norm=True,**kdeplot_args)
            if var == 'Area':
                continue # only plot once.

        if var == 'height': # plot the heights from the topog.
            kwargs = kdeplot_args.copy()
            kwargs.update(color='purple',linewidth=1,linestyle='dashed',label='_Topog')
            t=topog.where(topog > 0).stack(idx=['x', 'y']).dropna('idx')
            sns.kdeplot(t,common_norm=True,ax=ax, **kwargs)

        ax.set_xlabel(xlabel,fontsize='small')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        labels.plot(ax)
from matplotlib.ticker import LogLocator

locator = LogLocator(numticks=5)  # 5 ticks
for ax in axs.flat:
    if ax.get_xscale() == 'log':
        ax.xaxis.set_major_locator(locator)
        #ax.set_xlim(10,None)
    else:
        ax.set_xlim(0,None)
    if ax.get_yscale() == 'log':
        ax.yaxis.set_major_locator(locator)
        # truncate at 10^-3
        ylim = ax.get_ylim()
        ax.set_ylim(np.max([1e-3,ylim[0]]),None)

axs[1][2].legend(fontsize='x-small',ncols=2,loc='upper left',columnspacing=0.1,borderaxespad=0,borderpad=0.0)
fig.show()
commonLib.saveFig(fig)

## plot hot - cold for mid quantile
kde_kw = dict(common_norm=True,log_scale=(None,10),clip=(15,None))
datatset_1h_05=radar_dataset.sel(quantv=0.5,resample_prd='1h')
q_temp = datatset_1h_05.temp.quantile([0.25,0.5,0.75])
hot = datatset_1h_05.max_value.where(datatset_1h_05.temp > q_temp[2],drop=True)
hot = hot.where(hot > 15,drop=True)
cold = datatset_1h_05.max_value.where(datatset_1h_05.temp < q_temp[0],drop=True)
cold = cold.where(cold > 15,drop=True)
fig,ax = plt.subplots(nrows=1,ncols=1,clear=True,num='hot_cold_delta',figsize=(6,4))
sns.kdeplot(hot,ax=ax,label='Hot',color='red',**kde_kw)
sns.kdeplot(cold,ax=ax,label='Cold',color='blue',**kde_kw)
fig.show()
commonLib.saveFig(fig)
print(f'Medians hot:{float(hot.median()):4.1f} cold:{float(cold.median()):4.1f}')
with np.printoptions(precision=1,suppress=True):
    print(f'10-90% Hot: {np.percentile(hot,[10,90])} Cold: {np.percentile(cold,[10,90])}')

