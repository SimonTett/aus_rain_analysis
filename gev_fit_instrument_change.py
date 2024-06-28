# gev fits for various radar changes. A bit ad hoc and done as neeed.
import pathlib

import numpy as np

import ausLib
import xarray
from ausLib import comp_radar_fit
import typing
import scipy.stats
import pathlib

def output_values(fit: xarray.Dataset, fit_ref:xarray.Dataset,cov:str,title_str:str):

    print(f"Output values for {title_str}")

    for p in ['location', 'scale']:
        dp = f'D{p}_{cov}'
        dfract = fit.Parameters.sel(parameter=dp) / fit_ref.Parameters.sel(parameter=p)
        q = dfract.quantile([0.1, 0.5, 0.9], dim='sample').to_dataframe().unstack()
        print(f"fract {dp}: {q.round(3)}")
    # print out the AIC
    q_aic = (fit.AIC - fit_ref.AIC).quantile([0.1, 0.5, 0.9], dim='sample').to_dataframe().unstack()
    print(f"AIC: {q_aic.round(3)}")
    # 2 use the mean covariance matrix.
    cov_mean = fit.Cov.mean('sample')
    p_mean = fit.Parameters.mean('sample')
    samp_params = []
    for resample in p_mean.resample_prd:
        dist = scipy.stats.multivariate_normal(p_mean.sel(resample_prd=resample),
                                               cov_mean.sel(resample_prd=resample)
                                               )
        dist_sample = dist.rvs(size=100)
        dist_sample = xarray.DataArray(data=dist_sample,
                                       coords=dict(sample=np.arange(0, 100), parameter=p_mean.parameter)
                                       )
        dist_sample = dist_sample.assign_coords(resample_prd=resample).rename('cov_param_samp')
        samp_params.append(dist_sample)
    samp_params = xarray.concat(samp_params, dim='resample_prd')
    for p in ['location', 'scale']:
        dp = f'D{p}_{cov}'
        dfract = samp_params.sel(parameter=dp) / samp_params.sel(parameter=p)
        q = dfract.quantile([0.1, 0.5, 0.9], dim='sample').to_dataframe().unstack()
        print(f"fract -cov  {dp}: {q.round(3)}")


cols_want=['band','doppler','dp','eth_dhz_threshold','postchange_start','prechange_end']

for name in ['Melb_rain','Canberra_rain_melbourne','Sydney_rain_melbourne']:
    fit_dir = ausLib.data_dir / f'radar/processed/{name}/fits'
    event_data = fit_dir/f'../events_{name}_DJF.nc'
    radar_dataset = xarray.load_dataset(event_data)
    fit_t = xarray.load_dataset(fit_dir/"gev_fit_temp.nc")
    fit = xarray.load_dataset(fit_dir/"gev_fit.nc")
    meta = ausLib.site_info(ausLib.site_numbers[fit.attrs['site']])
    anomT=(radar_dataset.ObsT - fit_t.attrs['mean_temp']).rename('Tanom')
    print(meta.loc[:,cols_want])
    # melbourne has two changes -- C-> S band in 2007 and dual-polarisation in 2017
    if name == 'Melb_rain':

        s_c = xarray.where(radar_dataset.t < meta.iloc[0].loc['prechange_end'], 0, 1).rename('Sband')
        dp = xarray.where(radar_dataset.t > meta.iloc[1].loc['prechange_end'], 1, 0).rename('dp')
        fit_c = comp_radar_fit(radar_dataset, cov=[s_c],  n_samples=100,file=fit_dir/"gev_fit_Sband.nc")
        fit_dp = comp_radar_fit(radar_dataset, cov=[dp],  n_samples=100,file=fit_dir/"gev_fit_dp.nc")
        fit_c_dp = comp_radar_fit(radar_dataset, cov=[s_c,dp],  n_samples=100,file=fit_dir/"gev_fit_Sband_dp.nc")
        fit_dp_t = comp_radar_fit(radar_dataset, cov=[dp,anomT],  n_samples=100,file=fit_dir/"gev_fit_dp_T.nc")
        fit_c_dp_t = comp_radar_fit(radar_dataset, cov=[s_c,dp,anomT],  n_samples=100,file=fit_dir/"gev_fit_Sband_dp_T.nc")
        fit_c_t = comp_radar_fit(radar_dataset, cov=[s_c,anomT],  n_samples=100,file=fit_dir/"gev_fit_Sband_T.nc")
        output_values(fit_dp_t,fit_dp,'Tanom',f'{name} DP+T vs DP')
    elif any([name.startswith('Canberra_rain'),name.startswith('Sydney_rain')]): # canbera & Sydney have one change -- addition of doppler.
        doppler = xarray.where(radar_dataset.t < meta.iloc[0].loc['prechange_end'], 0, 1).rename('doppler')
        fit_doppler = comp_radar_fit(radar_dataset, cov=[doppler],  n_samples=100,file=fit_dir/"gev_fit_doppler.nc")
        fit_doppler_t = comp_radar_fit(radar_dataset, cov=[doppler,anomT],  n_samples=100,file=fit_dir/"gev_fit_doppler_T.nc")
        output_values(fit_doppler_t,fit_doppler,'Tanom',f'{name} Doppler+T vs Doppler')