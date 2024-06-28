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

commonLib.init_log(my_logger,level='INFO')
site = 'Melbourne'
name='Melb_rain'
site = 'Sydney'
name = 'Sydney_rain_melbourne'
radar_dataset = xarray.load_dataset(ausLib.data_dir/f"radar/processed/{name}/events_{name}_DJF.nc") # load the processed radar events
#radar_dataset = radar_dataset.where(radar_dataset.t.dt.year > 2000) # drop the first years.
topog_file = list((ausLib.data_dir/f'radar/site_data/{site}').glob('*_cbb_dem.nc'))[0]
topog_cbb = xarray.load_dataset(topog_file).coarsen(x=4, y=4, boundary='trim').mean()
topog = topog_cbb.elevation
cbb = topog_cbb.CBB
topog = topog.where(cbb <= 0.5)# mask values where CBB > 0.5

my_logger.debug(f"Loaded datasets")
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=[8, 8],clear=True,
                        num=f'kde_smooth_events_{name}',sharex='col', sharey='col',layout='constrained')

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
                        ['Solar Hour','Area (km$^2$)','Height (m)','Mx Rain (mm/h)'],
                        [ dict(gridsize=24,clip=(0,24)), # hours
                          dict(gridsize=50,log_scale=(10,None)), # area
                          dict(gridsize=50,clip=(0,1000),log_scale=(10,None)), # ht
                          dict(gridsize=50,clip=(0,None),log_scale=(None,10)) # rain
                          ]):

        for prd,color in zip(radar_quant.resample_prd.values[0:-1],colors): # don't want the last resample prd.
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



