import pathlib
import unittest

import numpy as np
import pandas as pd
import xarray
import ausLib
import numpy.testing as nptest
import matplotlib.pyplot as plt # so when need to visualise can easly do so!
test_dir = pathlib.Path("test_data")
class test_ausLib(unittest.TestCase):

    def notest_process_radar(self):
        ds = xarray.load_dataset(test_dir/"28_19980924_rainrate.nc")

        dataSet = ausLib.process_radar(ds)
        # test data is 1 day. So expect to have only one field in the output.
        self.assertEqual(dataSet.time.size,1)
        # expect max >= mean.
        null = dataSet.max_rain.isnull() & dataSet.mean_rain.isnull() #
        self.assertTrue(np.all((dataSet.max_rain >= dataSet.mean_rain).where(~null,True)))
        # test time_max_rain mask matches null.
        self.assertTrue(np.all(dataSet.time_max_rain.isnull() == null))
        # test that count_rain_thresh <= count_rain
        msk = ~dataSet.count_1h_thresh.isnull()
        nptest.assert_array_equal(~msk,null) # masks should be the same.
        self.assertTrue(np.all(dataSet.count_1h_thresh.where(msk,0.0) <= dataSet.count_1h))
        rain_1h = ds.rainrate.resample(time='1h').mean()
        # apply sufficiency filter!
        m2 = ds.isfile.resample(time='1h').mean() > 0.8
        rain_1h = rain_1h.where(m2)
        cnt_1h_thresh = (rain_1h > 1).sum('time',min_count=1).squeeze()
        cnt_1h_thresh = cnt_1h_thresh.where(msk) # mask out places that should be missing
        nptest.assert_array_equal(cnt_1h_thresh,dataSet.count_1h_thresh.squeeze())
        # test mean_< mean_thresh
        nptest.assert_array_less(dataSet.mean_rain,dataSet.mean_rain_thresh+0.1)

    def test_summary_process(self):
        ds = xarray.load_dataset(test_dir/"28_19980924_rainrate.nc")

        dataSet = ausLib.summary_process(ds,mean_resample='1h')
        # expect max >= mean.
        null = dataSet.max_rain.isnull() & dataSet.mean_rain.isnull() #
        self.assertTrue(np.all((dataSet.max_rain >= dataSet.mean_rain).where(~null,True)))
        # test time_max_rain mask matches null.
        self.assertTrue(np.all(dataSet.time_max_rain.isnull() == null))
        # test that count_rain_thresh <= count_rain
        msk = ~dataSet.count_1h_thresh.isnull()
        nptest.assert_array_equal(~msk,null) # masks should be the same.
        self.assertTrue(np.all(dataSet.count_1h_thresh.where(msk,0.0) <= dataSet.count_1h))
        rain_1h = ds.rainrate.fillna(0.0).resample(time='1h').mean()
        # apply sufficiency filter!
        m2 = ds.isfile.resample(time='1h').mean() > 0.8
        rain_1h = rain_1h.where(m2)
        cnt_1h_thresh = (rain_1h > 1).sum('time',min_count=1).squeeze()
        nptest.assert_array_equal(cnt_1h_thresh,dataSet.count_1h_thresh.squeeze())
        # test mean_< mean_thresh where have mean_thresh_values
        msk_thresh = ~dataSet.mean_rain_thresh.isnull()
        nptest.assert_array_less(dataSet.mean_rain.where(msk_thresh),dataSet.mean_rain_thresh+0.1)

    def test_read_gdsp_metadata(self):
        # test read_gdsp_metadata
        series = ausLib.read_gsdp_metadata(test_dir/"AU_088162.txt")
        self.assertIsInstance(series,pd.Series)
        # assert the name is the ID.
        self.assertEqual(series.name,series.ID)
        # assert have 21 lines in the header.
        self.assertEqual(series.header_lines,21)

    def test_read_gsdp_data(self):
        # test that read_data works
        meta_data = ausLib.read_gsdp_metadata(test_dir/"AU_088162.txt")
        series = ausLib.read_gsdp_data(meta_data)
        # len of series should be as expected from meta-data
        nrec= meta_data.loc['Number of records']
        self.assertEqual(len(series),nrec)
        # and fraction missing should be as expected.
        percent_missing = float(100*series.isnull().sum())/nrec
        self.assertAlmostEqual(percent_missing,meta_data.loc['Percent missing data'],places=2)

    def test_gsdp_metadata(self):
        # test gsdp_metadata
        files=test_dir.glob("AU_*.txt")
        meta_data = ausLib.gsdp_metadata(files)
        self.assertIsInstance(meta_data,pd.DataFrame)
        # can't think of other checks...

    def test_process_gsdp_record(self):
        # test process_gsdp_record
        meta_data = ausLib.read_gsdp_metadata(test_dir / "AU_088162.txt")
        series = ausLib.read_gsdp_data(meta_data)
        process = ausLib.process_gsdp_record(series)
        # test max >= mean
        self.assertTrue((process.max_rain >= process.mean_rain).all())


if __name__ == '__main__':
    unittest.main()
