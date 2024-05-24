# little code to test parts of process_reflectivity
import typing
import multiprocessing

from process_reflectivity import *
import ausLib
import logging

if __name__ == "__main__":
    my_logger = ausLib.my_logger
    ausLib.init_log(my_logger, 'DEBUG')
    multiprocessing.freeze_support()  # needed for obscure reasons I don't get!
    client = ausLib.dask_client()
    #dask.config.set(scheduler="single-threaded")  # make sure dask is single threaded.
    coarsen = dict(x=4, y=4)
    coarsen_method: typing.Literal['mean', 'median'] = 'median'
    # read in info
    zip_file = pathlib.Path('/g/data/rq0/hist_gndrefl/2/2020/2_20201125.gndrefl.zip')
    drop_vars_first = ['error', 'reflectivity']
    fld_info = read_zip(zip_file, drop_variables=drop_vars_first, coarsen=coarsen,
                                         first_file=True,parallel=True)

    my_logger.info('Reading in first file')
    drop_vars = ['error', 'x_bounds', 'y_bounds', 'proj']
    ds = read_zip(zip_file, coarsen=coarsen, concat_dim='valid_time', drop_variables=drop_vars,
                  parallel=True, coarsen_method=coarsen_method,
                  combine='nested', chunks=dict(valid_time=6), engine='netcdf4')
    ds = ds.merge(fld_info)
    my_logger.info('Reading in second file')
    zip_file = pathlib.Path('/g/data/rq0/hist_gndrefl/2/2020/2_20201222.gndrefl.zip')
    ds2 = read_zip(zip_file, coarsen=coarsen, concat_dim='valid_time', drop_variables=drop_vars,
        parallel=True, coarsen_method=coarsen_method,
                   combine='nested', chunks=dict(valid_time=6), engine='netcdf4', max_value=85.0)
    ds2 = ds2.merge(fld_info)


