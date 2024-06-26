# plot the meta data from the ppi files
import matplotlib.pyplot as plt
import numpy as np
import xarray
import pathlib
import ausLib

my_logger = ausLib.my_logger
# get the metadata in
metadata = dict()
for site in ausLib.site_numbers.keys():  # loop over keys
    root_dir = ausLib.data_dir / 'site_data' / f'{site}_metadata'
    if not root_dir.exists():
        my_logger.warning(f"No data for {site} at {root_dir}")
        continue
    files = sorted(root_dir.glob(f'{site}_*_metadata.nc'))
    if len(files) == 0:
        my_logger.warning(f'No files found for {site} in {root_dir}')
        continue
    ds = xarray.open_mfdataset(files, combine='nested', concat_dim='time').sel(time=slice('1997-01-01','2023-12-01')).load()
    metadata[site] = ds
    ## various fixes.
    if 'elevation' in ds:
        ds['min_elevation'] = ds['elevation'].min('elevation_index')
        ds['n_azimuths'] = ds['azimuth'].notnull().sum('azimuth_index')
        ds['resoln'] = ds['time_coverage_resolution'].dt.total_seconds()
    my_logger.debug(f'Loaded data for {site} from {files}')

# get in the radar meta-data so can plot change
stns = ausLib.read_radar_file('meta_data/long_radar_stns.csv')

## get some summary info.
# resoln shifts from 360 to 600 seconds
for site, ds in metadata.items():
    L = (ds.resoln > 400 )& (ds.time > np.datetime64('2010-01-01'))
    if L.sum() > 0:
        times = ds.time.where(L, drop=True)
        print(f'{site} has resolution > 400 seconds from  {times[0].dt.strftime("%Y-%m-%d").values}')



## and now to plot it.
fig_dir = pathlib.Path('figures/ppi_metadata')
fig_dir.mkdir(exist_ok=True, parents=True)
plot_style = dict(color='black', linestyle='-', marker='o', markersize=5, linewidth=1, markevery=24)
vars_to_plot = ['radar_beam_width_h', 'altitude', 'frequency', 'resoln', 'min_elevation', 'n_azimuths']
for v in vars_to_plot:
    fig, axes = ausLib.std_fig_axs(f'ppi_{v}', sharex=True)
    for site, ds in metadata.items():
        records = stns.loc[stns['id'] == ausLib.site_numbers[site]]
        if v in ds.data_vars:
            ds[v].plot(ax=axes[site], **plot_style)
            axes[site].tick_params(axis='x', labelsize='x-small',labelrotation=45)
            axes[site].tick_params(axis='y', labelsize='x-small')
            axes[site].set_ylabel('')
            axes[site].set_xlabel('')
            #axes[site].xaxis.locator_params(nbins=5)
            lim = np.array(axes[site].get_ylim())
            y = 0.75*lim[1]+0.25*lim[0]
            # add on breakpoints
            for name, r in records.iterrows():
                x=np.datetime64(r['postchange_start'])
                axes[site].axvline(x, color='red', linestyle='--')
                axes[site].text(x,y,r.radar_type,ha='left',va='center',fontsize='x-small',rotation=90)

        axes[site].set_title(site)
    fig.suptitle(v.capitalize())
    fig.show()
    fig.savefig(fig_dir / f'{v}.png')
    my_logger.info(f'Saved {fig_dir / f"{v}.png"}')
