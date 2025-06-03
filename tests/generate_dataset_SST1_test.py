import unittest
import torch as th
from pathlib import Path
import tempfile
import shutil
import calendar

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from generate_dataset_SST1 import CutImages

class TestCutImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory
        cls.test_dir = tempfile.mkdtemp()
        
        # Create sample data files for testing
        cls.create_test_data()
    
    @classmethod
    def tearDownClass(cls):
        # Remove the temporary directory after tests
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def create_test_data(cls):
        # Create sample nan mask files and data files
        cls.nan_masks_dir = Path(cls.test_dir) / "nan_masks"
        cls.nan_masks_dir.mkdir()
        cls.data_dir = Path(cls.test_dir) / "data"
        cls.data_dir.mkdir()
        
        # Create sample files for 4 months
        cls.days_per_month = {"2020_01": 31, "2020_02": 29, "2020_03": 31, "2020_04": 30, 
                              "2020_11": 30, "2023_02": 28, "2023_11": 30}
        cls.months = ["2020_01", "2020_02", "2020_03", "2020_04", "2020_11", "2023_02", "2023_11"]
        for month in cls.months:
            # Nan masks (31 days, 100x100)
            nan_mask = th.ones((31, 100, 100), dtype=th.bool)
            # Add some random nans (about 10%)
            nan_mask = th.where(th.rand_like(nan_mask.float()) < 0.9, nan_mask, ~nan_mask)
            th.save(nan_mask, cls.nan_masks_dir / f"{month}.pt")
            
            # Data files (31 days, 4 channels, 100x100)
            # Channel 0: SST, Channel 1: stdev, Channel 2: lat, Channel 3: lon
            data = th.rand((31, 4, 100, 100))
            th.save(data, cls.data_dir / f"{month}.pt")
    
    def setUp(self):
        # Common parameters for testing
        self.original_nrows = 100
        self.original_ncols = 100
        self.final_nrows = 32
        self.final_ncols = 32
        self.n_images = 10
        self.nans_threshold = 0.1
        self.surrounding_days = 3
        self.max_pixels = int(self.final_nrows * self.final_ncols * self.nans_threshold)
        self.nan_placeholder = -300
        self.params = {
            "dataset": {
                "dataset_kind": "test",
                "nan_placeholder": self.nan_placeholder,
                "cutted_nrows": self.final_nrows,
                "cutted_ncols": self.final_ncols,
                "n_cutted_images": self.n_images,
                "nans_threshold": self.nans_threshold,
                "total_days": 2 * self.surrounding_days + 1,
                "test": {
                    "n_rows": self.original_nrows,
                    "n_cols": self.original_ncols,
                }
            }
        }
        
    def test_initialization(self):
        """Test that initialization works with valid parameters"""
        cut = CutImages(self.params)
        
        self.assertEqual(cut.original_nrows, self.original_nrows)
        self.assertEqual(cut.original_ncols, self.original_ncols)
        self.assertEqual(cut.final_nrows, self.final_nrows)
        self.assertEqual(cut.final_ncols, self.final_ncols)
        self.assertEqual(cut.n_images, self.n_images)
        self.assertEqual(cut.nans_threshold, self.nans_threshold)
        self.assertEqual(cut.total_days, self.params["dataset"]["total_days"])
        self.assertEqual(cut.surrounding_days, self.surrounding_days)
        self.assertEqual(cut.max_pixels, int(self.final_nrows * self.final_ncols * self.nans_threshold))
    
    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors"""
        # Test invalid original dimensions
        with self.assertRaises(ValueError):
            CutImages(self.params, original_nrows=0)
        
        with self.assertRaises(ValueError):
            CutImages(self.params, original_ncols=0)
            
        # Test invalid final dimensions
        with self.assertRaises(ValueError):
            CutImages(self.params, final_nrows=0)
            
        with self.assertRaises(ValueError):
            CutImages(self.params, final_ncols=0)
            
        # Test final dimensions larger than original
        with self.assertRaises(ValueError):
            CutImages(self.params, final_nrows=self.original_nrows + 1)
            
        with self.assertRaises(ValueError):
            CutImages(self.params, final_ncols=self.original_ncols + 1)
            
        # Test invalid nans threshold
        with self.assertRaises(ValueError):
            CutImages(self.params, nans_perc=-0.1)
            
        with self.assertRaises(ValueError):
            CutImages(self.params, nans_perc=1.1)
            
        # Test invalid number of images
        with self.assertRaises(ValueError):
            CutImages(self.params, n_images=0)
            
        # Test invalid total days
        with self.assertRaises(ValueError):
            CutImages(self.params, total_days=0)
            
        with self.assertRaises(ValueError):
            CutImages(self.params, total_days=29)
    
    def test_select_random_points(self):
        """Test that random points are selected correctly"""
        cut = CutImages(self.params)
        n_points = 5
        points = cut._select_random_points(n_points)
        
        # Check shape
        self.assertEqual(len(points), n_points)
        self.assertIsInstance(points, list)
        for point in points:
            self.assertIsInstance(point, tuple)
            self.assertEqual(len(point), 2)
        
            # Check values are within bounds
            self.assertTrue(point[0] >= 0)
            self.assertTrue(point[0] <= cut.original_nrows - cut.final_nrows)
            self.assertTrue(point[1] >= 0)
            self.assertTrue(point[1] <= cut.original_ncols - cut.final_ncols)
    
    def test_get_available_days(self):
        """Test that available days are correctly identified"""
        cut = CutImages(self.params)
        
        file_names = ["2020_01.pt", "2023_02.pt", "2020_11.pt", "2023_11.pt"]
        dummy_dir = Path("dummy_dir")
        path_list = [dummy_dir / file_name for file_name in file_names]
        available_days = cut._get_available_days(path_list)
        
        # Check we have the right number of months
        self.assertEqual(len(available_days), len(file_names))
        
        expected_ndays = [31, 28, 30, 30]  # Jan, Feb, Nov
        
        i = 0
        for path in path_list:
            n_days = available_days[path]
            
            self.assertEqual(len(n_days), expected_ndays[i])
            
            expected_days = list(range(1, expected_ndays[i] + 1))
            
            for day in expected_days:
                self.assertIn(day, n_days)
            i += 1
    
    def test_map_random_points_to_days(self):
        """Test that random points are correctly mapped to days"""
        cut = CutImages(self.params, n_images=1000)
        
        path_list = list((self.nan_masks_dir).glob("*.pt"))
        path_list.sort()
        points_to_days = cut._map_random_points_to_days(path_list)
        
        # Check we have the right number of points
        total_points = sum(len(v) for v in points_to_days.values())
        self.assertEqual(total_points, cut.n_images)
        
        # Check that days are valid for each month
        for path, day_point_list in points_to_days.items():
            month = Path(path).stem
            year, month_num = map(int, month.split("_"))
            max_days = calendar.monthrange(year, month_num)[1]
            
            print(f"Month: {month}, Max days: {max_days}")
            for day, point in day_point_list:
                self.assertTrue(1 <= day)
                self.assertTrue(day <= max_days)
                self.assertEqual(len(point), 2)
    
    def test_get_valid_nan_mask(self):
        """Test that valid nan masks are correctly identified"""
        cut = CutImages(self.params, nans_perc=0.5, max_trials=10)  # Higher threshold for testing
        
        # Load one of the test nan masks
        test_path = list((self.nan_masks_dir).glob("*.pt"))[0]
        original_nan_mask = th.load(test_path)
        
        # Test with a valid point
        day = 1
        point = (10, 10)
        new_point, cutted_mask = cut._get_valid_nan_mask(original_nan_mask, day, point)
        
        self.assertEqual(cutted_mask.shape, (cut.final_nrows, cut.final_ncols))
        self.assertTrue(th.sum(~cutted_mask) <= cut.max_pixels)
        self.assertIsInstance(new_point, tuple)
        self.assertEqual(len(new_point), 2)
        # Check values are within bounds
        self.assertTrue(new_point[0] >= 0)
        self.assertTrue(new_point[0] <= cut.original_nrows - cut.final_nrows)
        self.assertTrue(new_point[1] >= 0)
        self.assertTrue(new_point[1] <= cut.original_ncols - cut.final_ncols)
        
        # Test with a very low threshold (should raise ValueError after trials)
        cut.max_pixels = 1

        with self.assertRaises(ValueError):
            cut._get_valid_nan_mask(original_nan_mask, day, point)
    
    def test_get_days_list(self):
        """Test that the days list is correctly generated"""
        
        total_days = 7
        center_day = 3
        cut = CutImages(self.params, total_days=total_days)
        
        # Test with a date in the middle of the month
        date_str = "2020_01_15"
        days_list = cut._get_days_list(date_str)
        
        # Should have 7 days total (3 before, 1 center, 3 after)
        self.assertEqual(len(days_list), total_days)
        self.assertEqual(days_list[center_day], date_str)  # Center day
        
        # Check the sequence is correct
        expected_dates = [
            "2020_01_12", "2020_01_13", "2020_01_14",  # Before
            "2020_01_15",                              # Center
            "2020_01_16", "2020_01_17", "2020_01_18"   # After
        ]
        self.assertEqual(days_list, expected_dates)
        
        # Test with a date near month start (should handle month boundaries)
        date_str = "2020_01_02"
        expected_dates = [
            "2019_12_30", "2019_12_31", "2020_01_01",  # Before
            "2020_01_02",                              # Center
            "2020_01_03", "2020_01_04", "2020_01_05"   # After
        ]
        days_list = cut._get_days_list(date_str)
        self.assertEqual(len(days_list), total_days)
        self.assertEqual(days_list[center_day], date_str)
        
        # Test with a date near month end (should handle month boundaries)
        date_str = "2020_02_28"
        expected_dates = [
            "2020_02_25", "2020_02_26", "2020_02_27",  # Before
            "2020_02_28",                              # Center
            "2020_02_29", "2020_03_01", "2020_03_02"   # After
        ]
        days_list = cut._get_days_list(date_str)
        self.assertEqual(len(days_list), total_days)
        self.assertEqual(days_list[center_day], date_str)
    
    def test_get_data_paths_to_dataset_idx_dict(self):
        """Test the main dictionary creation method"""
        cut = CutImages(self.params, n_images=1000)
        path_list = list((self.nan_masks_dir).glob("*.pt"))
        path_list.sort()
        
        final_dict, nan_mask_tensor = cut.get_data_paths_to_dataset_idx_dict(path_list, self.data_dir)
        
        keys = list(final_dict.keys())
        
        # Check tensor shape
        self.assertEqual(nan_mask_tensor.shape, (cut.n_images, cut.final_nrows, cut.final_ncols))
    
        # Check dictionary structure
        # self.assertEqual(len(final_dict), len(path_list))  # One entry per month file
        
        # Check total points matches n_images
        total_points = sum(len(v) for v in final_dict.values())
        self.assertEqual(total_points, cut.n_images * cut.total_days)
        
        for key in keys:
            tuple_list = final_dict[key]
            # Check that each entry in the dictionary is a list of tuples
            self.assertIsInstance(tuple_list, list)
            max_days = self.days_per_month[self.data_dir / key.name]
            
            # print(f"Month: {key}, Max days: {max_days}")
            for entry in tuple_list:
                self.assertIsInstance(entry, tuple)
                self.assertEqual(len(entry), 3)
                self.assertIsInstance(entry[0], int)
                self.assertTrue(1 <= entry[0])
                self.assertTrue(entry[0] <= max_days)
                self.assertIsInstance(entry[1], tuple)
                self.assertEqual(len(entry[1]), 2)
                self.assertIsInstance(entry[2], tuple)
                self.assertEqual(len(entry[2]), 2)
        
    def test_cut_method(self):
        """Test the main cutting method"""
        cut = CutImages(self.params, n_images=20)
        path_list = list((self.nan_masks_dir).glob("*.pt"))
        data_paths = list((self.data_dir).glob("*.pt"))
        
        # First create the dictionary
        final_dict, _ = cut.get_data_paths_to_dataset_idx_dict(path_list, self.data_dir)
        
        
        
        # Then test the cut method
        cutted_images = cut.cut(final_dict, data_paths)
        
        # Check output shape
        n_channels = cut.total_days + 4  # days + (stdev, lat, lon, time)
        self.assertEqual(cutted_images.shape, (cut.n_images, n_channels, cut.final_nrows, cut.final_ncols))
        
        # Check time encoding is applied correctly
        time_channel = cutted_images[:, -1, :, :]
        self.assertTrue(th.all(time_channel[0] == time_channel[0, 0, 0]))  # All values same in channel
        self.assertTrue(-1 <= time_channel[0, 0, 0] <= 1)  # Cosine value
    
    def test_get_encoded_time(self):
        """Test that time encoding produces valid cosine values"""
        cut = CutImages(self.params)
        
        # Test with various days
        for day in [1, 15, 31]:
            encoded = cut._get_encoded_time(day)
            self.assertTrue(-1 <= encoded <= 1)
        
        # Test that different days give different encodings
        encodings = [cut._get_encoded_time(day) for day in range(1, 32)]
        self.assertEqual(len(set(encodings)), 31)  # All unique

if __name__ == "__main__":
    unittest.main()