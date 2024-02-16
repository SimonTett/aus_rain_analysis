import pathlib
import unittest

import numpy as np
import pandas as pd
import xarray
import ausLib

test_dir = pathlib.Path("test_data")
class test_ausLib(unittest.TestCase):

    def test_process_radar(self):
        dataSet = xarray.open_mfdataset(test_dir/"28_19980924_rainrate.nc")
        # test is to just run the code!
        dataSet = ausLib.process_radar(dataSet)
        # test data is 1 day. So expect to have only one field in the output.
        self.assertEqual(dataSet.time.size,1)
        # expect max >= mean.
        null = dataSet.max_rain.isnull() & dataSet.mean_rain.isnull() #
        self.assertTrue(np.all((dataSet.max_rain >= dataSet.mean_rain).where(~null,True)))
        # test time_max_rain mask matches null.
        self.assertTrue(np.all(dataSet.time_max_rain.isnull() == null))

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
