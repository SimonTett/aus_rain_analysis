#!/usr/bin/env python

"""tests for process_extract_metadata."""
import pathlib
import sys
import unittest

import numpy as np
import xarray

import process_extract_metadata


import ausLib

class TestProcessExtractMetadata(unittest.TestCase):
    """Empty test scaffolding for process_extract_metadata."""
    def setUp(self):
        self.example_level_1_zip = ausLib.level1_dir/"46/2021/vol/46_20210101.pvol.zip"
        self.example_ppi_zip = ausLib.level1b_dir/"46/ppi/2021/46_20210101_ppi.zip"
        if not self.example_level_1_zip.exists():
            raise ValueError(f"Example level 1 zip file {self.example_level_1_zip} does not exist.")

        if not self.example_ppi_zip.exists():
            raise ValueError(f"Example PPI zip file {self.example_ppi_zip} does not exist.")


    @unittest.skip("TODO: implement test")
    def test_month_sort(self):
        """Test month_sort path ordering helper. Not bothering with for now."""


    def test_extract_convert(self):
        """Test metadata extraction and basic type conversion."""
        dt:xarray.DataTree = ausLib.read_radar_zipfile(self.example_level_1_zip, first_file=True,datatree=True,
                                       file_pattern='*.h5')

        attrs=dt['dataset1/data1/how'].attrs
        result = process_extract_metadata.extract_convert(attrs)
        expected={}
        for variab,value in attrs.items():

            if isinstance(value,(np.ndarray,list)) and len(value) > 0:
                expected[variab+"_count"] = len(value)
                expected[variab+"_max"] = np.max(value)
                expected[variab+"_min"] = np.min(value)
            else:
                expected[variab] = value
        self.assertEqual(result,expected)

    def test_read_level1_meta(self):
        """Test level1 metadata extraction from pvol zip input."""

        ds = process_extract_metadata.read_level1_meta(self.example_level_1_zip)
        # expect to only two coords time and rapic_DBZLVL_index
        self.assertEqual(tuple(ds.dims),('time','rapic_DBZLVL_index'))
        # expect that rapic_CLEARAIR is bool.
        self.assertEqual(ds.rapic_CLEARAIR.dtype, 'bool')
        self.assertEqual(ds.rapic_CLEARAIR, False)
        expected_datavars = {'a1gate', 'antgainH', 'astart', 'beamwH', 'beamwV', 'beamwidth', 'elangle', 'height',
        'highprf', 'lat', 'level1_file', 'lon', 'nbins', 'nrays', 'peakpwrH', 'polmode', 'pulsewidth', 'rapic_AZCORR',
        'rapic_CLEARAIR', 'rapic_DBZCOR', 'rapic_DBZLVL','rapic_DBZLVL_count', 'rapic_DBZLVL_max', 'rapic_DBZLVL_min',
        'rapic_ELCORR','rapic_NOISETHRESH',  'rapic_VIDRES',  'rscale', 'rstart', 'scan_count', 'scan_index',
                             'wavelength'}
        self.assertEqual(set(ds.data_vars), expected_datavars)
    @unittest.skip("TODO: implement test")
    def test_iso_to_timedelta64(self):
        """Test ISO-8601 duration parsing helper. Not bothered with for noe"""


    def test_read_ppi_meta(self):
        """Test PPI metadata extraction from ppi zip input."""

        meta_data = process_extract_metadata.read_ppi_meta(self.example_ppi_zip)
        ds = ausLib.read_radar_zipfile(self.example_ppi_zip, first_file=True,concat_dim='time')
        expected_data_vars = {'azimuth', 'elevation', 'elevation_min','elevation_max','elevation_count',
                              'calibration_offset',
                              'instrument_name', 'time_coverage_resolution', 'time_coverage_duration','instrument_id',
                                          'origin_altitude','origin_latitude','origin_longitude',
                              'time_coverage_resolution', 'time_coverage_duration','ppi_file'}
        # expect to only two coords time and elevation_index
        self.assertEqual(dict(meta_data.sizes), {'time':1   ,'elevation_index':10})
        self.assertEqual(set(meta_data.data_vars), expected_data_vars) # and only these data vars.
        # check drop_variables is set correctly.
        expected_drop= ['time_coverage_start','time_coverage_end','time_reference',
                          'path_integrated_attenuation', 'sweep_number', 'sweep_start_ray_index',
                          'sweep_end_ray_index', 'sweep_mode', 'radar_beam_width_h', 'radar_beam_width_v',
                          'latitude', 'longitude', 'altitude', 'volume_number']
        self.assertEqual(process_extract_metadata.drop_variables,
                         expected_drop)
        # run again and should be no change.
        ldrop = len(process_extract_metadata.drop_variables)
        process_extract_metadata.read_ppi_meta(self.example_ppi_zip)
        self.assertEqual(process_extract_metadata.drop_variables,expected_drop)

    def test_process_extract_metadata(self):
        """
        Run the process_extract_metadata.py script and verify output.
        Returns:

        """
        import subprocess
        import tempfile
        direct = pathlib.Path(__file__).parent
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run([sys.executable, f"{direct}/process_extract_metadata.py", "Adelaide",
                            "--years","2021", "--outdir",tmpdir,"--glob","0101.pvol.zip"],check=True)

            self.assertEqual(result.returncode,0)
            output_file = pathlib.Path(tmpdir) / "Adelaide_2021_metadata.nc"
            self.assertTrue(output_file.exists())
            ds = xarray.load_dataset(output_file)
            ds.close()
        # check dimensions matter
        self.assertEqual(dict(ds.sizes),{'time':1,'rapic_DBZLVL_index':159,'elevation_index':10})

        expected_datavars = {'a1gate', 'antgainH', 'astart', 'beamwH', 'beamwV', 'beamwidth', 'elangle', 'height',
        'highprf', 'lat', 'level1_file', 'lon', 'nbins', 'nrays', 'peakpwrH', 'polmode', 'pulsewidth', 'rapic_AZCORR',
        'rapic_CLEARAIR', 'rapic_DBZCOR', 'rapic_DBZLVL','rapic_DBZLVL_count', 'rapic_DBZLVL_max', 'rapic_DBZLVL_min',
        'rapic_ELCORR','rapic_NOISETHRESH',  'rapic_VIDRES',  'rscale', 'rstart', 'scan_count', 'scan_index',
                             'wavelength', # level 1 data to here. Level 1b data below.
         'azimuth', 'elevation', 'elevation_min','elevation_max','elevation_count',
          'calibration_offset',
          'instrument_name', 'time_coverage_resolution', 'time_coverage_duration','instrument_id',
                      'origin_altitude','origin_latitude','origin_longitude',
          'time_coverage_resolution', 'time_coverage_duration','ppi_file'} # level1b
        # expect to only two coords time and elevation_index
        self.assertEqual(set(ds.data_vars),expected_datavars)


if __name__ == "__main__":
    unittest.main()
