# little code to test parts of process_reflectivity

from process_submit.process_reflectivity import *
import ausLib

if __name__ == "__main__":
    my_logger = ausLib.my_logger
    ausLib.init_log(my_logger, 'DEBUG')
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!
    client = ausLib.dask_client()
    #dask.config.set(scheduler="single-threaded")  # make sure dask is single threaded.
    pat = '2_2020121*.zip'
    outpath = 'test.nc'
    if ausLib.hostname.startswith('gadi'): # oz super-computer
        files = list((ausLib.hist_ref_dir/'2/2020').glob(pat))
    else:
        files =list((ausLib.data_dir/'test_data').glob(pat))
    files = sorted(files)
    ds = read_multi_zip_files(files, dbz_ref_limits=(15, 55),
                              coarsen=dict(x=4, y=4), coarsen_method= 'mean',coarsen_cv_max=10.0)
    attr_var = ds.drop_vars(['reflectivity', 'reflectivity_speckle', 'error'], errors='ignore'). \
        mean('time', keep_attrs=True)
    ref_summ = summary_process(ds.reflectivity, mean_resample=['30min','1h'], threshold=0.5, base_name='reflectivity')
    attr_var = attr_var.expand_dims(time=ref_summ.time)
    ref_summ = ref_summ.merge(attr_var)
    min_res = ref_summ.sample_resolution.dt.seconds/60.
    # check sample resolution is reasonable
    if min_res < 4:
        ValueError(f'small sample resolution {min_res} mins')
    elif min_res > 15:
        ValueError(f'large sample resolution {min_res} mins')

    ref_summ.encoding.update(zlib=True, complevel=4) # so data compressed on output.
    ref_summ.time.assign_attrs(units='minutes since 1990-01-01T00:00') # common time units. Needed by ncrcat.
    my_logger.info(f"computed month of summary data {memory_use()}")
    my_logger.info(f'Writing summary data to {outpath} {ausLib.memory_use()}')
    ref_summ.to_netcdf(outpath, unlimited_dims='time')



