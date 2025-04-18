import unittest
import torch as th
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from dataset_normalization import MinMaxNormalization

class TestMinMaxNormalization(unittest.TestCase):
    def setUp(self):
        
        dataset_shape = (100, 5, 128, 128)
        self.sample_images = th.randn(dataset_shape[0], dataset_shape[1], dataset_shape[2], dataset_shape[3])
        self.sample_masks = th.ones(dataset_shape[0], dataset_shape[1], dataset_shape[2], dataset_shape[3])
        
        masked_idxs = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 2, 2, 0], [0, 3, 3, 0], [0, 4, 4, 0]]
        for idx in masked_idxs:
            self.sample_masks[idx[0], idx[1], idx[2], idx[3]] = 0
            
        self.normalizer = MinMaxNormalization(batch_size=10)
        
    def test_normalize_small_dataset(self):
        """Test normalization on a small dataset."""
        
        images = self.sample_images[:5]
        masks = self.sample_masks[:5]
        norm_images, min_max = self.normalizer.normalize(images, masks)
        
        masks = masks.to(th.bool)
        self.assertEqual(norm_images.shape, images.shape)
        self.assertEqual(len(min_max), 2)
        self.assertTrue(th.all(norm_images[masks] >= 0))
        self.assertTrue(th.all(norm_images[masks] <= 1))
        self.assertTrue(th.all(norm_images[~masks] == images[~masks]))
        
    def test_normalize_large_dataset(self):
        """Test normalization on a large dataset."""
        
        images = self.sample_images
        masks = self.sample_masks
        norm_images, min_max = self.normalizer.normalize(images, masks)
        
        masks = masks.to(th.bool)
        self.assertEqual(norm_images.shape, images.shape)
        self.assertEqual(len(min_max), 2)
        self.assertTrue(th.all(norm_images[masks] >= 0))
        self.assertTrue(th.all(norm_images[masks] <= 1))
        self.assertTrue(th.all(norm_images[~masks] == images[~masks]))
    
    def test_denormalize(self):
        """Test denormalization."""
        
        images = self.sample_images
        masks = self.sample_masks
        norm_images, _ = self.normalizer.normalize(images, masks)
        
        minmax = [
            th.stack([self.sample_images[:, c].min() for c in range(self.sample_images.shape[1])]),
            th.stack([self.sample_images[:, c].max() for c in range(self.sample_images.shape[1])])
        ]
        
        # Denormalize
        denorm_images = self.normalizer.denormalize(norm_images, minmax)
        
        # Check if denormalized images are equal to original images where masks are 1
        masks = masks.to(th.bool)
        self.assertTrue(th.allclose(denorm_images[masks], images[masks], atol=1e-5))
            
        
        