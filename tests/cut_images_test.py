import unittest
import torch as th
from pathlib import Path
import tempfile
import os
import sys
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from generate_dataset import CutImages
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
                "dataset_name": "test",
                "mask_kind": self.mask_kind
            },
            "masks": {
                "square": {
                   "mask_percentage": self.mask_percentage,
                },
             },
            "test": {
                "n_rows": self.x_shape_raw,
                "n_cols": self.y_shape_raw,
                "channels_to_keep": [0, 1, 2, 3]
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
            dummy_date = f"{years.item()}_{months.item()}"
            th.save(dummy_image, self.processed_data_dir / f"{dummy_date}.pt")
            
        self.mask_function = SquareMask(self.params)
            
        self.cut_class = CutImages(self.params)

    def tearDown(self):
        """Clean up temporary directory after tests."""
        self.temp_dir.cleanup()
        
    def test_get_available_days(self):
        """Test that the get_available_days method returns the correct list of days."""
        path_list = [Path("path_to_folder/2023_01.pt"), Path("path_to_folder/2025_02.pt")]
        expected_days = {path_list[0]: [day for day in range(1, 32)],
                         path_list[1]: [day for day in range(1, 29)]}
        
        days = self.cut_class._get_available_days(path_list)
        
        # Check that the output is a dictionary with the correct keys and values
        self.assertIsInstance(days, dict)
        self.assertEqual(len(days), len(path_list))
        self.assertSetEqual(set(days.keys()), set(path_list))
        for month in path_list:
            self.assertSetEqual(set(days[month]), set(expected_days[month]))
            
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
            
    def test_calculate_time_interval(self):
        """Test that the calculate_time_interval method returns the correct time interval."""
        path_to_idx_map = {
            "path_to_folder/2023_01.pt": [(2, [1, 2]), (3, [1, 2])],
            "path_to_folder/2025_02.pt": [(1, [1, 2]), (12, [1, 2])]
        }
        
        oldest_date, newest_date, interval = self.cut_class._calculate_time_interval(path_to_idx_map)
        
        # Check that the output is a tuple with the correct values
        self.assertIsInstance(oldest_date, datetime.date)
        self.assertIsInstance(newest_date, datetime.date)
        self.assertIsInstance(interval, int)
        self.assertEqual(oldest_date, datetime.date(2023, 1, 2))
        self.assertEqual(newest_date, datetime.date(2025, 2, 12))
        self.assertEqual(interval, 772)
        
    def test_select_random_points(self):
        """Test that select_random_points returns the correct number of points."""
        
        random_points = self.cut_class.select_random_points(n_points=self.n_cutted_images)
        # Check that the number of points is correct
        self.assertEqual(len(random_points), self.n_cutted_images)
        # Check that the points are within the bounds of the original image
        for point in random_points:
            self.assertTrue(0 <= point[0] < self.x_shape_raw)
            self.assertTrue(0 <= point[1] < self.y_shape_raw)
    
    def test_generate_cutted_images(self):
        n_channels = 4
        
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
            
    # def test_generate_masked_datasets_keys(self):
    #     """Test that generate_masked_image_dataset returns a dictionary with the correct keys."""
    #     self.assertIsInstance(self.dataset, dict)
    #     # Check that the output is a dictionary with the correct keys
    #     self.assertSetEqual(set(self.dataset.keys()), {"images", "masks"})

    # def test_generate_masked_datasets_shapes_and_dtypes(self):
    #     """Test that generate_masked_image_dataset returns tensors with the correct shapes and dtypes."""
        
    #     # Check that each tensor has the correct shape and dtype
    #     expected_shape = (self.n_cutted_images, self.n_channels, self.cutted_nrows, self.cutted_ncols)
            

    #     self.assertIsInstance(self.dataset["images"], th.Tensor)
    #     self.assertEqual(self.dataset["images"].shape, expected_shape)
    #     self.assertEqual(self.dataset["images"].dtype, th.float32)
        
    #     self.assertIsInstance(self.dataset["masks"], th.Tensor)
    #     self.assertEqual(self.dataset["masks"].shape, expected_shape)
    #     self.assertEqual(self.dataset["masks"].dtype, th.bool)
            
    #     self.assertIsInstance(self.nans_mask, th.Tensor)
    #     self.assertEqual(self.nans_mask.shape, expected_shape)
    #     self.assertEqual(self.nans_mask.dtype, th.bool)
        
    # def test_generate_masked_datasets_values(self):
    #     """Test that generate_masked_image_dataset returns tensors with the correct masked channels."""
        
    #     # Check that the masked channels have 0's and the other channels have all 1's
    #     keys = list(self.dataset.keys())
    #     for i in range(self.n_cutted_images):
    #         for j in self.masked_channels:
    #             self.assertTrue((self.dataset[keys[1]][:, j, :, :] == 0).any())
        
    #     # Check that the mask has no nan
    #     mask_key = list(self.dataset.keys())[1]
    #     self.assertFalse(th.isnan(self.dataset[mask_key]).any(), f"NaNs found in dataset_min[{mask_key}]")
        
    #     # Check that the nan values are replaced with the placeholder value
    #     for i in range(self.n_cutted_images):
    #         for j in range(self.n_channels):
    #             nan_mask = self.nans_mask[i, j, :, :]
    #             image = self.dataset[keys[0]][i, j, :, :]
    #             self.assertTrue(th.all(image[nan_mask == 0] == self.nan_placeholder), "Not all values under the nan mask are placeholder")
                
    # def test_nans_coverage(self):
    #     """Test that the masks cover all the nans in the images."""
    #     # Check that the 0s in the masks of dataset_min[keys[1]] cover all the nans in dataset_min[keys[0]]
        
    #     keys = list(self.dataset.keys())
    #     for i in range(self.n_cutted_images):
    #         nan_mask = self.nans_mask[i]
    #         mask = self.dataset[keys[1]][i]
            
    #         # Check that the mask covers all the nans
    #         self.assertTrue(th.all(mask[nan_mask == 0] == 0), "Mask does not cover all nans")
            
    # def test_same_mask(self):
    #     """Test that the same mask is used for all images."""
    #     # Check that the masks are the same for all images
    #     keys = list(self.dataset.keys())
    #     masks = self.dataset[keys[1]]
        
    #     for i in range(1, self.n_cutted_images):
    #         image_mask = masks[i, self.masked_channels[0], :, :]
    #         for channel in range(0, self.n_channels):
    #             # Check that the masked channels are the same
    #             if channel in self.masked_channels:
    #                 self.assertTrue(th.all(image_mask == masks[i, channel, :, :]), f"Masked channels are not the same for each masked channel")
    #             else:
    #                 # Check that the non masked channels are not masked
    #                 self.assertTrue(th.all(masks[i, channel, :, :] == 1), f"Non masked channel has 0s in the mask")
     
if __name__ == "__main__":
    unittest.main()