import unittest
import torch as th
from pathlib import Path
import tempfile
import os
import sys
import datetime
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from generate_dataset_SST import CutImages
from utils.mask_data import SquareMask


class TestGenerateMaskedImageDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.processed_data_dir = Path(self.temp_dir.name)

        # Create dummy processed images
        self.n_images = 5
        self.x_shape_raw = 30
        self.y_shape_raw = 40
        self.n_channels = 4
        # Test parameters
        self.n_cutted_images = 8
        self.cutted_nrows = 10
        self.cutted_ncols = 15
        self.mask_percentage = 0.5
        self.masked_channels = [0, 2]
        self.nan_placeholder = -2.0
        self.nans_threshold = 0.5
        self.mask_kind = "square"
        
        # Create sample json file
        self.params = {
            "dataset": {
                "cutted_nrows": self.cutted_nrows,
                "cutted_ncols": self.cutted_ncols,
                "n_cutted_images": self.n_cutted_images,
                "nans_threshold": self.nans_threshold,
                "dataset_kind": "test",
                "mask_kind": self.mask_kind,
                "test": {
                    "n_rows": self.x_shape_raw,
                    "n_cols": self.y_shape_raw,
                    "channels_to_keep": [0, 1, 2, 3]
                    }
            },
            "masks": {
                "square": {
                   "mask_percentage": self.mask_percentage,
                },
             }
        }
        
        n_days = 31
        
        # Create dummy input with nans
        for i in range(self.n_images):
            # Create a random image
            dummy_image = th.rand(n_days, self.n_channels - 1, self.x_shape_raw, self.y_shape_raw)
            
            # Select random points to mark as nan
            random_x = th.randint(0, self.x_shape_raw, (1,))
            random_y = th.randint(0, self.y_shape_raw, (1,))
            
            # Create a mask with 0s in the selected points
            nan_mask = th.ones(n_days, self.n_channels - 1, self.x_shape_raw, self.y_shape_raw)
            nan_mask[:, self.masked_channels, random_x, random_y] = 0
            
            # Set the selected points to nan
            dummy_image[nan_mask == 0] = th.nan
            
            # Save the image with the correct name
            years = th.randint(2000, 2020, (1,))
            months = th.randint(1, 13, (1,))
            dummy_date = f"{years.item()}_{months.item():02d}"
            th.save(dummy_image, self.processed_data_dir / f"{dummy_date}.pt")
            
        self.mask_function = SquareMask(self.params)
            
        self.cut_class = CutImages(self.params)

    def tearDown(self):
        """Clean up temporary directory after tests."""
        self.temp_dir.cleanup()
        
    def test_get_encoded_time(self):
        days = [1, 2, 3, 4, 5]
        days_per_year = 365.25
        
        encoded_times = [math.cos((day / days_per_year) * 2 * math.pi) for day in days]
        
        # Check that the encoded times are correct
        for i, day in enumerate(days):
            self.assertAlmostEqual(self.cut_class._get_encoded_time(day), encoded_times[i], places=5)
            
    def test_map_random_points_to_days(self):
        """Test that the map_random_points_to_days method returns the correct mapping."""
        path_list = [Path("path_to_folder/2023_01.pt"), Path("path_to_folder/2025_02.pt")]
        days_list = {
            path_list[0]: [day for day in range(1, 32)],
            path_list[1]: [day for day in range(1, 29)]
            }
        
        points_list = [[0, 0], [1, 1], [2, 2], [3, 3]]
        
        point_map = self.cut_class.map_random_points_to_days(days_list=days_list, points_list=points_list)
        
        # Check that the output is a dictionary with the correct keys and values
        self.assertIsInstance(point_map, dict)
        # Check that all keys of point_map are in the keys of days_list
        for day in list(point_map.keys()):
            self.assertIn(day, days_list.keys())
        
        # Check that all points in points_list appear exactly once in the dictionary
        all_points = []
        for (day, points) in point_map.items():
            for p in points:
                all_points.append(p[1])
        
        self.assertEqual(len(all_points), len(points_list))
        for point in points_list:
            self.assertIn(point, all_points)
        
    def test_select_random_points(self):
        """Test that select_random_points returns the correct number of points."""
        
        random_points = self.cut_class.select_random_points(n_points=self.n_cutted_images)
        # Check that the number of points is correct
        self.assertEqual(len(random_points), self.n_cutted_images)
        # Check that the points are within the bounds of the original image
        for point in random_points:
            self.assertTrue(0 <= point[0] < self.x_shape_raw)
            self.assertTrue(0 <= point[1] < self.y_shape_raw)
    
    def test_cut_valid_image(self):
        n_channels = 4
        height = 30
        width = 40
        img_shape = (n_channels, height, width)
        raw_image = th.ones(img_shape) * th.nan
        idx = [0, 5]
        
        cut_class = CutImages(original_nrows=height, original_ncols=width, final_nrows = 10, final_ncols = 10, nans_threshold=0.5, n_cutted_images=3)
        cut_class.cutted_nrows = 10
        cut_class.cutted_ncols = 10
        
        with self.assertRaises(ValueError):
            cut_class._cut_valid_image(raw_image, idx, 100)
            
        raw_image = th.randn(img_shape)
        raw_image[:, 0:5, 0:5] = th.nan
        
        cutted_image = cut_class._cut_valid_image(raw_image, [2, 3], cut_class.cutted_nrows * cut_class.cutted_ncols)
        
        # Check that the cutted image has the correct shape
        self.assertEqual(cutted_image.shape, (n_channels, cut_class.cutted_nrows, cut_class.cutted_ncols))
        # Check that the cutted image preserves the nans of the original image
        self.assertTrue(th.all(th.isnan(cutted_image[:, 0:3, 0:2])))
    
    def test_generate_cutted_images(self):
        detected_files = list(Path(self.processed_data_dir).glob("*.pt"))
        available_days = self.cut_class._get_available_days(detected_files)
        random_points = self.cut_class.select_random_points(n_points=self.n_cutted_images)
        path_to_idx_map = self.cut_class.map_random_points_to_days(days_list=available_days, points_list=random_points)
        
        dataset, nan_masks = self.cut_class.generate_cutted_images(self.n_channels, path_to_idx_map, self.nan_placeholder)
        
        check_nan_mask = th.where(dataset == self.nan_placeholder, 0, 1).bool()
        
        self.assertIsInstance(dataset, th.Tensor, "The dataset is not a tensor")
        self.assertIsInstance(nan_masks, th.Tensor, "The nan masks are not a tensor")
        self.assertEqual(dataset.dtype, th.float32, "The dataset is not a float32 tensor")
        self.assertEqual(nan_masks.dtype, th.bool, "The nan masks are not a bool tensor")
        self.assertEqual(dataset.shape, (self.n_cutted_images, self.n_channels, self.cutted_nrows, self.cutted_ncols),
                            "The dataset does not have the correct shape")
        self.assertEqual(nan_masks.shape, (self.n_cutted_images, self.n_channels, self.cutted_nrows, self.cutted_ncols),
                            "The nan masks do not have the correct shape")
        self.assertEqual(th.isnan(dataset).sum(), 0, "The dataset has nans")
        self.assertTrue(th.all(check_nan_mask == nan_masks), "The nan masks are not the same as the dataset")
        
        
    def test_initialization(self):
        """Test if the CutAndMaskImage class initializes correctly."""
        # Test if the CutAndMaskImage class initializes correctly
        self.assertIsInstance(self.cut_class, CutImages)
        
        # Check if the parameters are set correctly
        self.assertEqual(self.cut_class.cutted_nrows, self.cutted_nrows)
        self.assertEqual(self.cut_class.cutted_ncols, self.cutted_ncols)
        self.assertEqual(self.cut_class.original_nrows, self.x_shape_raw)
        self.assertEqual(self.cut_class.original_ncols, self.y_shape_raw)
        self.assertEqual(self.cut_class.nans_threshold, self.nans_threshold)
        self.assertEqual(self.cut_class.n_cutted_images, self.n_cutted_images)
     
if __name__ == "__main__":
    unittest.main()