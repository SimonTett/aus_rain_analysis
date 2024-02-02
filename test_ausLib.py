import unittest
import xarray
import ausLib

class test_ausLib(unittest.TestCase):
    def test_something(self):
        dataSet = xarray.open_mfdataset("/g/data/rq0/level_2/28/RAINRATE/28_2000*_rainrate.nc")
        mean_resample = '1h'
        max_resample = '1MS'
        min_mean = 0.8
        min_max = 0.8
        time_dim = 'time'
        radar_range = 150e3
        # test is to just run the code!
        dataSet = ausLib.process_radar(dataSet)

        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
