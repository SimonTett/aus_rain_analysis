# plot GEV for shapes of -0.5, 0 and 0.5

import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import commonLib
from matplotlib.ticker import ScalarFormatter,LogLocator

fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(9,5),num='example_gev',clear=True,layout='constrained')
shapes = [-1,-0.5,0,0.5,1]
colors = ['purple','blue','orange','green','brown']
x = np.geomspace(1e-2,0.5,100)
for shape,color in zip(shapes,colors):
    gev = scipy.stats.genextreme(c=shape)
    axs.plot(1.0/x,gev.isf(x),color=color,label=f'shape={-shape}') # python convention different to Cole book
axs.set_xscale('log')
axs.set_yscale('log')
axs.set_xlabel('Return period ')
axs.set_ylabel('Return value')
axs.text(0.95,0.1,transform=axs.transAxes,s=r'$x=y* \text{scale} + \text{loc}$',ha='right',va='bottom')
# Increase number of ticks on the x-axis
axs.set_xticks([2,5,10, 20,50, 100])
axs.get_xaxis().set_major_formatter(ScalarFormatter())

axs.set_yticks(np.array([2,5,10, 20,50, 100])/10)
axs.get_yaxis().set_major_formatter(ScalarFormatter())

axs.legend()
fig.show()
commonLib.saveFig(fig)