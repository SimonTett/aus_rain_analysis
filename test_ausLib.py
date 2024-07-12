import datetime
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
        cnt_1h_thresh = (rain_1h > 0.1).sum('time',min_count=1).squeeze()
        nptest.assert_array_equal(cnt_1h_thresh,dataSet.count_1h_thresh.squeeze())
        # test mean_< mean_thresh where have mean_thresh_values
        msk_thresh = ~dataSet.mean_rain_thresh.isnull()
        nptest.assert_array_less(dataSet.mean_rain.where(msk_thresh),dataSet.mean_rain_thresh+0.1)

    def test_read_gdsr_metadata(self):
        # test read_gdsr_metadata
        series = ausLib.read_gsdr_metadata(test_dir / "AU_088162.txt")
        self.assertIsInstance(series,pd.Series)
        # assert the name is the ID.
        self.assertEqual(series.name,series.ID)
        # assert have 21 lines in the header.
        self.assertEqual(series.header_lines,21)

    def test_read_gsdr_data(self):
        # test that read_data works
        meta_data = ausLib.read_gsdr_metadata(test_dir / "AU_088162.txt")
        series = ausLib.read_gsdr_data(meta_data)
        # len of series should be as expected from meta-data
        nrec= meta_data.loc['Number of records']
        self.assertEqual(len(series),nrec)
        # and fraction missing should be as expected.
        percent_missing = float(100*series.isnull().sum())/nrec
        self.assertAlmostEqual(percent_missing,meta_data.loc['Percent missing data'],places=2)

    def test_gsdr_metadata(self):
        # test gsdr_metadata
        files=test_dir.glob("AU_*.txt")
        meta_data = ausLib.gsdr_metadata(files)
        self.assertIsInstance(meta_data,pd.DataFrame)
        # can't think of other checks...

    def test_process_gsdr_record(self):
        # test process_gsdr_record
        meta_data = ausLib.read_gsdr_metadata(test_dir / "AU_088162.txt")
        series = ausLib.read_gsdr_data(meta_data)
        process = ausLib.process_gsdr_record(series)
        # test max >= mean
        self.assertTrue((process.max_rain >= process.mean_rain).all())

    def test_utc_offset(self):
        import pytz
        # test utc_offset works.

        coords = dict( # each member of dict should be 3 element tuple in order, lng, lat & expected difference in hours
            Edinburgh = (-3.,56.0,0.0),
            canberra=(149.1289,-35.282,10.),
            Sydney=(151.21, -33.87, 10.0),
            Perth_Oz=(115.86,-31.956,8.0),
            # Australian exceptions
            Giles=(128.3, -25.0333,9.5),
            Broken_Hill = (141.467,-31.95,10.0),
            Eucla = (128.883,-31.675,8.)
                      )

        # test all co-ords
        for name,(lng,lat,delta) in coords.items():
            offset = ausLib.utc_offset(lng=lng,lat=lat)
            self.assertEqual(offset.seconds, delta * 3600.,msg=f'UTC offset failed for {name}')

    def test_index_ll_pts(self):
        # test index_ll works
        grid_lon,grid_lat = np.meshgrid(np.linspace(0,359,360),
                                        np.linspace(-90,90,181))
        pts=np.array([[0],[0]])
        indices = ausLib.index_ll_pts(grid_lon,grid_lat,pts)
        nptest.assert_array_equal(indices,([0],[90]))
        # try a bunch of pooints.
        xind = np.array([0,10,30,200])
        yind = np.array([0,10,40,180])
        xpts = grid_lon[0,xind]
        ypts = grid_lat[yind,0]
        pts = np.row_stack((xpts,ypts))
        indices = ausLib.index_ll_pts(grid_lon,grid_lat,pts)
        for indx,pt in zip(indices,[xind,yind]):
            nptest.assert_array_equal(pt,indx)
        # now try with tolerance
        pts[0,:] = pts[0,:]+0.1
        with self.assertRaises(ValueError):
            indices = ausLib.index_ll_pts(grid_lon, grid_lat, pts)
        indices = ausLib.index_ll_pts(grid_lon, grid_lat, pts,tolerance=0.2)
        for indx,pt in zip(indices,[xind,yind]):
            nptest.assert_array_equal(pt,indx)

    def test_comp_ratios(self):
        # test comp_ratios works
        # test data
        coords={'x':[0,1],'parameter':['location','Dlocation_Tanom','scale','Dscale_Tanom','shape']}
        params= xarray.DataArray(np.array([[1,2,3,4,5],[4,5,6,7,8]]),dims=coords.keys(),coords=coords)
        coords2={'x':[0,1],'parameter':['Dlocation_Tanom','Dscale_Tanom']}
        expected_results=xarray.DataArray(np.array([[2.0,4/3.],[5/4,7/6]]),dims=coords2.keys(),coords=coords2)
        results = ausLib.comp_ratios(params)
        self.assertTrue(np.all(results == expected_results))

if __name__ == '__main__':
    unittest.main()
