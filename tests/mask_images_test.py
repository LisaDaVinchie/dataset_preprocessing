import torch as th
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils.mask_data import mask_inversemask_image

class TestApplyMaskOnChannel(unittest.TestCase):
  def setUp(self):
    # Create a sample image tensor 
    self.images = th.tensor([
      [[[1, 2, 3],
        [4, 5, 6],
        [7, 8, th.nan]]],
      [[[10, 20, 30],
        [40, th.nan, 60],
        [70, 80, 90]]]
      ], dtype=th.float32)

    # Create a sample mask tensor 
    self.masks = th.tensor([
      [[[1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]]],
      [[[0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]]]
      ], dtype=th.float32)

  def test_placeholder_provided(self):
      """Test when a placeholder value is provided."""
      placeholder = -1.0
      masked_img, inverse_masked_img = mask_inversemask_image(self.images, self.masks, placeholder)

      # Expected masked image
      expected_masked_img1 = th.tensor([
        [[[1, placeholder, 3],
          [placeholder, 5, placeholder],
          [7, placeholder, placeholder]]]
        ], dtype=th.float32)

      expected_masked_img2 = th.tensor([
        [[[placeholder, 20, placeholder],
          [40, placeholder, 60],
          [placeholder, 80, placeholder]]]
        ], dtype=th.float32)

      # Expected inverse masked image
      expected_inverse_masked_img1 = th.tensor([
        [[[placeholder, 2, placeholder],
          [4, placeholder, 6],
          [placeholder, 8, placeholder]]]
        ], dtype=th.float32)

      expected_inverse_masked_img2 = th.tensor([
        [[[10, placeholder, 30],
          [placeholder, placeholder, placeholder],
          [70, placeholder, 90]]]
        ], dtype=th.float32)

      # Check masked images
      self.assertTrue(th.allclose(masked_img[0], expected_masked_img1))
      self.assertTrue(th.allclose(masked_img[1], expected_masked_img2))

      # Check inverse masked images
      self.assertTrue(th.allclose(inverse_masked_img[0], expected_inverse_masked_img1))
      self.assertTrue(th.allclose(inverse_masked_img[1], expected_inverse_masked_img2))

  def test_placeholder_none(self):
      """Test when placeholder is None (mean replacement)."""
      masked_img, inverse_masked_img = mask_inversemask_image(self.images, self.masks, placeholder=None)

      # Compute expected means for each image
      mean1 = (1 + 3 + 5 + 7) / 4
      mean2 = (20 + 40 + 60 + 80) / 4
      inv_mean1 = (2 + 4 + 6 + 8) / 4  # Mean of unmasked values in the first image
      inv_mean2 = (10 + 30 + 70 + 90) / 4  # Mean of unmasked values in the second image

      # Expected masked image
      expected_masked_img1 = th.tensor([
        [[[1, mean1, 3],
          [mean1, 5, mean1],
          [7, mean1, mean1]]]
        ], dtype=th.float32)

      expected_masked_img2 = th.tensor([
        [[[mean2, 20, mean2],
          [40, mean2, 60],
          [mean2, 80, mean2]]]
        ], dtype=th.float32)

      # Expected inverse masked image
      expected_inverse_masked_img1 = th.tensor([
        [[[inv_mean1, 2, inv_mean1],
          [4, inv_mean1, 6],
          [inv_mean1, 8, inv_mean1]]]
        ], dtype=th.float32)

      expected_inverse_masked_img2 = th.tensor([
        [[[10, inv_mean2, 30],
          [inv_mean2, inv_mean2, inv_mean2],
          [70, inv_mean2, 90]]]
        ], dtype=th.float32)

      # Check masked images
      self.assertTrue(th.allclose(masked_img[0], expected_masked_img1))
      self.assertTrue(th.allclose(masked_img[1], expected_masked_img2))

      # Check inverse masked images
      self.assertTrue(th.allclose(inverse_masked_img[0], expected_inverse_masked_img1))
      self.assertTrue(th.allclose(inverse_masked_img[1], expected_inverse_masked_img2))

  def test_all_masked(self):
      """Test when the entire image is masked."""
      masks_all_zero = th.zeros_like(self.masks)  # All pixels masked
      masked_img, inverse_masked_img = mask_inversemask_image(self.images, masks_all_zero, placeholder=None)

      # Expected result: All values replaced with the mean of the entire image
      mean1 = 0.0
      mean2 = 0.0
      
      inv_mean1 = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) / 8
      inv_mean2 = (10 + 20 + 30 + 40 + 60 + 70 + 80 + 90) / 8

      expected_masked_img1 = th.full_like(self.images[0], mean1)
      expected_masked_img2 = th.full_like(self.images[1], mean2)

      # Expected inverse masked image: Original image, since no inverse masking is applied
      expected_inverse_masked_img1 = self.images[0]
      expected_inverse_masked_img1[th.isnan(expected_inverse_masked_img1)] = inv_mean1
      expected_inverse_masked_img2 = self.images[1]
      expected_inverse_masked_img2[th.isnan(expected_inverse_masked_img2)] = inv_mean2

      # Check masked images
      self.assertTrue(th.allclose(masked_img[0], expected_masked_img1))
      self.assertTrue(th.allclose(masked_img[1], expected_masked_img2))

      # Check inverse masked images
      self.assertTrue(th.allclose(inverse_masked_img[0], expected_inverse_masked_img1))
      self.assertTrue(th.allclose(inverse_masked_img[1], expected_inverse_masked_img2))

  def test_all_unmasked(self):
      """Test when no pixels are masked."""
      masks_all_one = th.ones_like(self.masks)  # No pixels masked
      masked_img, inverse_masked_img = mask_inversemask_image(self.images, masks_all_one, placeholder=None)

      mean1 = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) / 8
      mean2 = (10 + 20 + 30 + 40 + 60 + 70 + 80 + 90) / 8
      
      # Expected masked image: original image, except for NaN values
      expected_masked_img1 = self.images[0]
      expected_masked_img1[th.isnan(expected_masked_img1)] = mean1
      expected_masked_img2 = self.images[1]
      expected_masked_img2[th.isnan(expected_masked_img2)] = mean2


      inv_mean1 = 0.0
      inv_mean2 = 0.0

      expected_inverse_masked_img1 = th.full_like(self.images[0], inv_mean1)
      expected_inverse_masked_img2 = th.full_like(self.images[1], inv_mean2)
      
      # Check masked images
      self.assertTrue(th.allclose(masked_img[0], expected_masked_img1))
      self.assertTrue(th.allclose(masked_img[1], expected_masked_img2))

      # Check inverse masked images
      self.assertTrue(th.allclose(inverse_masked_img[0], expected_inverse_masked_img1))
      self.assertTrue(th.allclose(inverse_masked_img[1], expected_inverse_masked_img2))

if __name__ == "__main__":
  unittest.main()