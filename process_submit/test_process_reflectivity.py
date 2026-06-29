#!/usr/bin/env python
"""
Test cases for process_reflectivity.

Prompt summary:
- Keep the original example code that reads a radar ZIP and processes it.
- Add a filesystem cache for expensive radar ZIP reads so repeated test runs
  can reuse previously loaded xarray datasets.
- Cache one ZIP file to one zipped Zarr file, keyed by the ZIP path, its mtime,
  and the reader kwargs.
- Use the cache for normal tests, but keep the dedicated IO test on the real
  read_radar_zip_file path.

  Tests writtin by SFBT using AI boilerplate. AI wrote caching layer.
"""


import hashlib
import json
import logging
import pathlib
import shutil
import tempfile
from contextlib import contextmanager
import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd
import xarray
import xarray.testing

import ausLib
import process_reflectivity
from zarr.storage import ZipStore



# Cache directory is intentionally hard-wired for now so the tests can reuse
# expensive radar reads across runs without needing extra configuration.
TEST_CACHE_DIR = pathlib.Path(__file__).with_name("_cache")
TEST_CACHE_DIR.mkdir(exist_ok=True)

# Use the module logger so cache activity shows up alongside the code under test.
log = process_reflectivity.my_logger
level = "INFO"
ausLib.init_log(log, level=level)
ausLib.init_log(process_reflectivity.my_logger, level=level)

# Keep the original reader around so the cache wrapper can call the real IO
# function even when tests temporarily monkeypatch process_reflectivity.
REAL_READ_RADAR_ZIP_FILE = process_reflectivity.read_radar_zip_file


def _normalise_for_json(value):
    """Convert common non-JSON values into a stable serialisable form.

    Args:
        value: A value pulled from the reader kwargs or cache metadata.

    Returns:
        A JSON-friendly representation that preserves the meaningful identity
        of the value for cache-key purposes.
    """
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, dict):
        return {str(key): _normalise_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_for_json(item) for item in value]
    return value


def _cache_key(zip_file: pathlib.Path, kwargs: dict) -> str:
    """Build a cache key from the input ZIP file state and reader kwargs.

    Args:
        zip_file: Path to the radar ZIP being read.
        kwargs: Keyword arguments passed through to
            process_reflectivity.read_radar_zip_file.

    Returns:
        A hex digest string used to name the cache entry.
    """
    payload = {
        "zip_file": str(zip_file.resolve()),
        "mtime_ns": zip_file.stat().st_mtime_ns,
        "kwargs": _normalise_for_json(kwargs),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_path(zip_file: pathlib.Path, kwargs: dict) -> pathlib.Path:
    """Return the path for the zipped Zarr cache entry.

    Args:
        zip_file: Path to the radar ZIP being read.
        kwargs: Keyword arguments passed through to the reader.

    Returns:
        The filesystem path of the corresponding zipped Zarr cache file.
    """
    return TEST_CACHE_DIR / f"{zip_file.stem}_{_cache_key(zip_file, kwargs)}.zarr.zip"


def read_radar_zip_file_cached(zip_file:pathlib.Path, **kwargs):
    """Cached wrapper for process_reflectivity.read_radar_zip_file.

    If the cache exists and is valid for the current ZIP mtime and kwargs, load
    that dataset instead of re-reading the ZIP archive. Otherwise read the real
    input and persist it to a zipped Zarr cache for the next test run.

    Args:
        zip_file: Radar ZIP file to read.
        **kwargs: Keyword arguments forwarded unchanged to
            process_reflectivity.read_radar_zip_file. These become part of the
            cache identity, so changing them produces a separate cache file.

    Returns:
        The loaded xarray.Dataset, or None if the underlying reader returns
        None.
    """
    zip_file = pathlib.Path(zip_file)
    cache_path = _cache_path(zip_file, kwargs)
    log.debug("cache lookup for %s with kwargs=%s", zip_file, kwargs)

    if cache_path.exists():
        log.debug("cache hit for %s", cache_path)
        log.info("reading cached radar dataset from %s", cache_path)
        with ZipStore(str(cache_path), mode="r") as store:
            return xarray.open_zarr(store).load()

    log.debug("cache miss for %s", cache_path)
    log.info("reading radar zip file from %s", zip_file)
    ds = REAL_READ_RADAR_ZIP_FILE(zip_file, **kwargs)
    if ds is None:
        log.debug("reader returned None for %s; nothing cached", zip_file)
        return None

    log.info("writing radar cache to %s", cache_path)
    with tempfile.TemporaryDirectory(dir=str(TEST_CACHE_DIR)) as tmpdir:
        zarr_dir = pathlib.Path(tmpdir) / "zarr"
        ds.to_zarr(zarr_dir, mode="w",zarr_format=2)
        tmp_base = cache_path.with_suffix("").with_name(cache_path.stem + ".tmp")
        tmp_archive = pathlib.Path(
            shutil.make_archive(str(tmp_base), "zip", root_dir=zarr_dir)
        )
        tmp_archive.replace(cache_path)

    return ds


@contextmanager
def use_cached_radar_reader():
    """Temporarily route process_reflectivity.read_radar_zip_file through cache.

    Tests that care about processing logic rather than IO can use this context
    manager to keep the same call shape while avoiding repeated ZIP reads.

    Yields:
        Nothing. While active, process_reflectivity.read_radar_zip_file points
        at read_radar_zip_file_cached and is restored afterwards.
    """
    log.debug("installing cached radar reader")
    process_reflectivity.read_radar_zip_file = read_radar_zip_file_cached
    try:
        yield
    finally:
        process_reflectivity.read_radar_zip_file = REAL_READ_RADAR_ZIP_FILE
        log.debug("restored real radar reader")

class TestProcessReflectivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):


        cls.sample_zip =ausLib.hist_ref_dir/"46/2021/46_20210101.gndrefl.zip"
        cls.sample_rainfield3 = ausLib.rainfields3_dir/"46/2024/46_20240101.gndrefl.zip"

        cls.sample_levels = np.array([0.0, 1.0, 2.0], dtype=float)
        cls.example_da = xarray.DataArray(
            np.array([[0.0, np.nan,-32],[0.2,1.0,12]], dtype=float),
            dims=("time","x"),
            coords={"time": np.array(["2000-01-01T00:10", "2000-01-01T00:20"], dtype="datetime64[ns]")
                    , "x": np.array([0.0, 1.0,2.0], dtype=float)},
        )
        cls.example_da.attrs['units'] = 'dBZ'
        cls.example_da.attrs['long_name'] = 'Example variable'
        cls.example_da.x.attrs['units'] = 'km'



    def test_empty_ds(self):
        # test empty_ds.
        # generate expected output.
        expected=dict()
        resample_prd = ['1h','2h']
        expected['var_float_1']=self.example_da.copy().rename('var_float_1').where(False)
        expected['var_float_2']=self.example_da.copy().rename('var_float_2').where(False).expand_dims(resample_prd=resample_prd)
        expected['var_time']=self.example_da.copy().rename('var_time').where(False).astype('<M8[ns]')
        expected = xarray.Dataset(expected)
        got = process_reflectivity.empty_ds(self.example_da,resample_prd,
                                            non_time_variables=['var_float_1','var_float_2'],
                                            vars_time=['var_time'], no_resample_vars=['var_time','var_float_1'])
        self.assertEqual(expected,got)





    def test_fix_spatial_units(self):
        ds = xarray.Dataset(dict(example=self.example_da.copy()))
        expected = (ds.x*1000.).assign_attrs(units='m')
        ds2 = process_reflectivity.fix_spatial_units(ds)
        self.assertEqual(ds2.x, expected)


    def test_quantize_to_levels(self):
        levels = [0.,1.0,2.]
        data = np.linspace(0,2.5,251)
        qdata = process_reflectivity.quantize_to_levels(data,levels)
        self.assertEqual(qdata.shape, data.shape)
        npt.assert_array_equal(np.unique(qdata), levels)
        expected = data.copy()
        expected[data<1] = 0.0
        expected[(data>=1) &( data<2.0)] = 1.0
        expected[data>=2.0] = 2.0
        npt.assert_array_equal(qdata, expected)



    def test_xarray_quantize(self):

        quant_levels = np.array([0.0, 1.0, 10.0], dtype=float)
        result = process_reflectivity.xarray_quantize(self.example_da, quant_levels)
        expected = self.example_da.copy()
        expected = expected.where(self.example_da>1.0,0)
        expected = expected.where((self.example_da > 10.0) | (self.example_da <1),1)
        expected = expected.where(self.example_da <= 10.0,10)
        expected = expected.where(self.example_da.notnull(),np.nan)
        result.attrs.clear()
        expected.attrs.clear()
        xarray.testing.assert_identical(result, expected)



    def test_read_radar_file(self):
        # This test must exercise the real IO path, not the cache.
        ds = process_reflectivity.read_radar_zip_file(self.sample_zip,concat_dim='valid_time',first_file=True,
                                                    region=dict(x=slice(0,100),y=slice(0,100)))
        self.assertIsInstance(ds, xarray.Dataset)
        self.assertIn('valid_time', ds.coords)
        self.assertEqual(ds.valid_time.size, 1) # only read first field, so should be 1.
        self.assertEqual(ds.x.size, 200) # 100 km with 500m boxes.
        self.assertEqual(ds.y.size, 200)
        self.assertEqual(ds.x.attrs['units'], 'm')
        self.assertEqual(ds.y.attrs['units'], 'm')
        # monotonically increasing.
        self.assertTrue((ds.x.values[1:] > ds.x.values[:-1]).all())
        self.assertTrue((ds.y.values[1:] > ds.y.values[:-1]).all())
        self.assertEqual(set([d for d in ds.data_vars]),
                         {'reflectivity','proj','x_bounds','y_bounds','error','count_reflectivity_missing'})


    def test_process_radar_file(self):
        # Process a cached raw dataset so repeated test runs avoid ZIP IO.
        with use_cached_radar_reader():
            raw = process_reflectivity.read_radar_zip_file(self.sample_zip,concat_dim='valid_time')

        ds = process_reflectivity.process_radar_file(raw, (0.0272, 0.653), dbz_ref_limits=(0, 55.), check_finite=True)
        # expect to have more nans than raw as get nans from values > upper dbz_ref_limits
        expected_count_nan = int(raw.reflectivity.isnull().sum() + (raw.reflectivity > 55.0).sum())
        expected_count_zero = int((raw.reflectivity < 0.0).sum())
        count_nan = int(ds.rain_rate.isnull().sum())
        count_zero = int((ds.rain_rate<=0.0).sum())
        self.assertEqual(expected_count_nan, count_nan)
        self.assertEqual(expected_count_zero, count_zero)

        # check that the rain rate is in the expected range.
        # max rain should be 0.0272*(10**5.5)**0.653
        max_rain = 0.0272 * ((10 ** 5.5) ** 0.653)  # this is about 100 mm/hr
        min_rain = 0.0272 # min rain possible is 0.0272*1/16 boxes
        max_value = ds.rain_rate.max()
        self.assertTrue(max_value <= max_rain)
        self.assertTrue(ds.rain_rate.min() == 0)
        # min_value > 0 should be >= 0.0272 b as lower limit is 0 dBZ.
        min_value = ds.rain_rate.where(ds.rain_rate > 0).min()
        self.assertTrue(min_value >= min_rain)
        print(
            f"min_value: {min_value:.3f} min_rain {min_rain:.3f} max_value: {max_value:.1f} max_rain {max_rain:.1f} mm/hr")

        ds = process_reflectivity.process_radar_file(raw, (0.0272, 0.653), dbz_ref_limits=(0, 55.), check_finite=True, coarsen=dict(x=4, y=4))
        # Expect to have 150 x 150 points after coarsening from 500m to 2km over 150x150km box
        self.assertEqual(ds.x.size, 150)
        self.assertEqual(ds.y.size, 150)
        min_rain = 0.0272 / 16  # min rain possible is 0.0272*1/16 boxes
        min_value = ds.rain_rate.where(ds.rain_rate > 0).min()
        self.assertTrue(min_value >= min_rain)

        # check quantize  works. WIll put in a silly one so everything gets set to 0  10 or 40.
        qlevels = np.array([0,10,40])
        ds2 = process_reflectivity.process_radar_file(raw, (0.0272, 0.653), check_finite=True, quantize_levels=qlevels)
        # convert levels to rain rate.
        qlevels = np.unique(np.append(qlevels, -32))
        qq=qlevels-float(raw.attrs['calibration_estimate']) # quantisation happens in uncorrected data. Calibration_est removed from data
        qq = np.clip(qq,-32.,None)
        qrain = np.append((10**(np.log10(0.0272)+(qq/10)*0.653)),np.nan)
        npt.assert_array_equal(np.unique(ds2.rain_rate), qrain)
        # now deal with quantization from metadata. More tricky. Get 2020's Adelaide calibration.
        files = (ausLib.data_dir / 'site_data/Adelaide_metadata').glob('Adelaide_202*_metadata.nc')
        files = sorted(files)
        metadata = xarray.open_mfdataset(files,combine='nested',concat_dim='time',coords='minimal')
        qlevels = metadata['rapic_DBZLVL'].load()
        ds3 = process_reflectivity.process_radar_file(raw, (0.0272, 0.653), check_finite=True, quantize_levels=qlevels)
        # work out expected values.
        qlev_base = qlevels.sel(time=raw.valid_time.values[0], method='nearest',tolerance=pd.Timedelta('16D'))
        qlev = qlev_base[qlev_base.notnull()]
        qlev = np.append(qlev - float(raw.attrs['calibration_estimate']),-32.0)
        qrain = 10**(np.log10(0.0272)+(qlev/10)*0.653)
        unique_rain = np.unique(ds3.rain_rate)
        unique_rain = unique_rain[np.isfinite(unique_rain)] # remove nans
        in_expected = np.isin(unique_rain,qrain)
        for v in unique_rain[~in_expected]:
            print(f"unexpected rain value {v} ")
            idx_close = np.abs(v-qrain).argmin()
            print(f"closest match is {float(qrain[idx_close])} at index {int(idx_close)} abs diff: {float(np.abs(v-qrain[idx_close]))}")

        self.assertTrue(np.all(in_expected))

        # test with calibration value. Should fail.
        with self.assertRaises(ValueError):
            ds = process_reflectivity.process_radar_file(raw, (0.0272, 0.653), dbz_ref_limits=(15, 55.), check_finite=True,
                                                         coarsen=dict(x=4, y=4),calibration=2.0)
        # read in a rainfields3 file and test correction works.
        with use_cached_radar_reader():
            raw_rf3 = process_reflectivity.read_radar_zip_file(self.sample_rainfield3,concat_dim='valid_time')
        calib = float(metadata.calibration_offset.sel(time=raw_rf3.valid_time.values[0], method='nearest',tolerance=pd.Timedelta('16D')))
        ds = process_reflectivity.process_radar_file(raw_rf3, (0.0272, 0.653), dbz_ref_limits=(15., 55.),
                                                     calibration=calib,check_finite=True, coarsen=dict(x=4, y=4))
        # Check  No calibration differs by a factor of 10**(calib*0.653) for all values. Zeros will stay zero.
        # need to adjust the dBZ limits for the tests to take account of the calibration so we have an easy comparison.
        ds_check = process_reflectivity.process_radar_file(raw_rf3, (0.0272, 0.653), dbz_ref_limits=(15+calib, 55.+calib),
                                                     check_finite=True, coarsen=dict(x=4, y=4))
        scale = 10.**(-calib*0.653/10.0)
        xarray.testing.assert_allclose(ds.rain_rate,ds_check.rain_rate*scale)
        # check get same results with metadata.

        ds2 = process_reflectivity.process_radar_file(raw_rf3, (0.0272, 0.653), dbz_ref_limits=(15., 55.),
                                                     calibration=metadata.calibration_offset,check_finite=True, coarsen=dict(x=4, y=4))
        xarray.testing.assert_allclose(ds.rain_rate,ds2.rain_rate)













    def test_read_multi_zip_files(self):
        zip_files = sorted(self.sample_zip.parent.glob("*.zip"))[0:4] # just want 4 files

        # read_multi_zip_files calls the ZIP reader internally, so wrap the call
        # to reuse cached per-file datasets instead of redoing the IO.
        with use_cached_radar_reader():
            ds = process_reflectivity.read_multi_zip_files(zip_files,(0.0272, 0.653),dbz_ref_limits=(0, 55.),
                                                           coarsen=dict(x=4, y=4),
                                                           region=dict(x=slice(-50,50),y=slice(-50,50)))
        # Expect to have 100 x 100 points after coarsening from 500m to 2km over 100 km box
        self.assertEqual(ds.x.size, 50)
        self.assertEqual(ds.y.size, 50)
        #

        self.assertEqual(ds.ancil_time.size,1) # should be 1 ancil time for all files.
        self.assertTrue(ds.time.size > 0.9*24*12*len(zip_files)) #  N days, 5-minute data (24*12) and at least 90% there.

        for var in ['proj','x_bounds','y_bounds']:
            self.assertIn(var, ds.data_vars)
        for var in ['longitude','latitude','x','y','time','ancil_time']:
            self.assertIn(var, ds.coords)
        self.assertEqual(set(ds.proj.dims),{'ancil_time'})
        self.assertEqual(set(ds.longitude.dims),{'y','x','ancil_time'})
        self.assertEqual(set(ds.latitude.dims),{'y','x','ancil_time'})
        self.assertEqual(set(ds.x_bounds.dims), {'x', 'bounds','ancil_time'})
        self.assertEqual(set(ds.y_bounds.dims), {'y', 'bounds','ancil_time'})


    def test_summary_process(self):
        import warnings

        warnings.filterwarnings("error", category=RuntimeWarning,
                                module=r"^xarray(\.|$)",
                                   #module=r"^wradlib(\.|$)",
                                )

        # read in several files to check that the summary process works
        zip_files = sorted(self.sample_zip.parent.glob("*202101*.zip")) # a whole month of data.
        # get example dataset -- just for first time.
        with use_cached_radar_reader():
            example  = process_reflectivity.read_radar_zip_file(self.sample_zip,concat_dim='valid_time',first_file=True,)

        # get maeta data to test specifying levels.
        ds = xarray.open_dataset(ausLib.data_dir / 'site_data/Adelaide_metadata/Adelaide_2021_metadata.nc')

        levels = ds.rapic_DBZLVL.sel(time=example.valid_time, method='nearest')
        log.info(f"Q levels at time {levels.time}")
        levels = levels.where(levels.notnull(), drop=True).values
        levels = np.append(levels, np.nan)
        levels = levels[np.isfinite(levels)]
        levels = np.insert(levels, 0,-32.0) # bottom value is -32 dBZ

        to_rain = (0.0272, 0.653)
        with use_cached_radar_reader():
            month = process_reflectivity.read_multi_zip_files(zip_files,to_rain,dbz_ref_limits=(0, 55.),
                                                           coarsen=dict(x=4, y=4), quantize_levels=levels,#
                                                           region=dict(x=slice(-50,50),y=slice(-50,50)))
        # expected values
        time_dim = 'time'
        resample_dict = {'time': '1h', 'closed': 'right', 'label': 'right'}
        rain = month.rain_rate
        bad = rain.isnull().all(time_dim, keep_attrs=True)  # find out where *all* data null
        mean_resamp = rain.resample(**resample_dict).mean(skipna=True)
        count  = rain[time_dim].resample(**resample_dict).count()
        ok =( count >= 6.0)
        mean_resamp = mean_resamp.where(ok,drop=True) # enough samples to get a good result.


        max_mean_resamp = mean_resamp.max(time_dim, keep_attrs=True, skipna=True).where(~bad,drop=True)
        time_max_resamp = mean_resamp.idxmax(time_dim, keep_attrs=False, skipna=True).where(~bad,drop=True)
        median_mean_resamp = mean_resamp.median(time_dim, keep_attrs=False, skipna=True).where(~bad,drop=True)
        mean_mean_resamp = mean_resamp.mean(time_dim, keep_attrs=False, skipna=True).where(~bad,drop=True)
        # now to run the processing.
        processed_ds = process_reflectivity.summary_process(month.rain_rate,base_name='rain_rate',min_fract_avg=0.5)
        # these are all the results of processing the first month of data.
        # verify the max 1h at a point is correct.
        indx = dict(x=30,y=30)

        for var_name,var in zip(['max_rain_rate','median_rain_rate','mean_rain_rate'],[max_mean_resamp,median_mean_resamp,mean_mean_resamp]):
            got = processed_ds[var_name].sel(resample_prd='1h').squeeze('time',drop=True)
            try:
                npt.assert_allclose(got.values,var.values,err_msg=f"Failed to compare for {var_name}",atol=5e-7) # do not check dims and within fp tolerance
            except AssertionError as e:
                print(f"Failed to compare for {var_name}")

                max_loc = np.abs(got-var).argmax(...)
                max_loc= {k:int(v) for k,v in max_loc.items()}
                print(f"max_loc: {max_loc}")
                print(f"got: {got.isel(**max_loc).values} Expected: {var.isel(**max_loc).values}")
                raise e








    def test_mask_anaprop(self):
        self.skipTest("Boilerplate only; add assertions for mask_anaprop")


if __name__ == '__main__':
    unittest.main()
