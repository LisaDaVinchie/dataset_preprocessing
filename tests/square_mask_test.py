import unittest
from tempfile import NamedTemporaryFile
import os
import sys
import json
import torch as th
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils.mask_data import SquareMask

class TestSquareMask(unittest.TestCase):
    
    def setUp(self):
        self.image_nrows = 10
        self.image_ncols = 15
        self.mask_percentage = 0.10
        
        self.params = {
            "masks": {
                "square": {
                    "mask_percentage": self.mask_percentage
                }
            },
            "dataset": {
                "cutted_nrows": self.image_nrows,
                "cutted_ncols": self.image_ncols
            }
        }
        
        # Create a LinesMask instance
        self.lines_mask = SquareMask(params=self.params)
    
    def test_initialization(self):
        """Test if the LinesMask class initializes correctly."""
        self.assertEqual(self.lines_mask.image_nrows, self.image_nrows)
        self.assertEqual(self.lines_mask.image_ncols, self.image_ncols)
        self.assertEqual(self.lines_mask.mask_percentage, self.mask_percentage)
        
    def test_invalid_parameters(self):
        """Test if the LinesMask class raises errors for invalid parameters."""
        with self.assertRaises(ValueError):
            SquareMask(params=self.params, image_nrows=-1)
        
        with self.assertRaises(ValueError):
            SquareMask(params=self.params, image_ncols=-1)
        
        with self.assertRaises(ValueError):
            SquareMask(params=self.params, mask_percentage=1.5)
        
        with self.assertRaises(ValueError):
            SquareMask(params=self.params, mask_percentage=-0.5)
        with self.assertRaises(ValueError):
            SquareMask(params=self.params, image_nrows=0)
        with self.assertRaises(ValueError):
            SquareMask(params=self.params, image_ncols=0)
        with self.assertRaises(ValueError):
            SquareMask(params=self.params, mask_percentage=0)
    
    def test_mask_shape(self):
        """Test if the mask shape is correct."""
        mask = self.lines_mask.mask()
        self.assertEqual(mask.shape, (self.image_nrows, self.image_ncols))
        
    def test_mask_dtype(self):
        """Test if the mask dtype is correct."""
        mask = self.lines_mask.mask()
        self.assertEqual(mask.dtype, th.bool)
        

if __name__ == '__main__':
    unittest.main()