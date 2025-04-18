import torch as th
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def apply_mask_on_channel(images: th.Tensor, masks: th.Tensor, placeholder: float = None) -> th.Tensor:
    """Mask the image with the mask, using a placeholder. If the placeholder is none, use the mean of the level"""
    new_images = images.clone()
    if placeholder is not None:
        return new_images * masks + placeholder * (1 - masks)
    
    means = (images * masks).sum(dim=(2, 3), keepdim=True) / (masks.sum(dim=(2, 3), keepdim=True))
    return new_images * masks + means * (1 - masks)

def mask_inversemask_image(images: th.Tensor, masks: th.Tensor, placeholder: float = None) -> tuple:
    """Mask the image with a placeholder value.
    If the placeholder is none, use the mean of the level and, if there are nans, mask them as well.

    Args:
        images (th.Tensor): images to mask, of shape (batch_size, channels, nrows, ncols)
        masks (th.Tensor): masks to apply, of shape (batch_size, channels, nrows, ncols)
        placeholder (float, optional): number to use for masked pixels. Defaults to None.

    Returns:
        tuple: masked_images, inverse masked images
    """
    new_images = images.clone()
    
    masks[images.isnan()] = 0
    inverse_masks = 1 - masks
    inverse_masks[images.isnan()] = 0
    
    new_images[new_images.isnan()] = 0
    
    if placeholder is None:
        placeholder, inv_placeholder = 0, 0
        
        if masks.sum() > 0:
            placeholder = (new_images * masks).sum(dim=(2, 3), keepdim=True) / (masks.sum(dim=(2, 3), keepdim=True))
        
        if inverse_masks.sum() > 0:
            inv_placeholder = (new_images * inverse_masks).sum(dim=(2, 3), keepdim=True) / (inverse_masks.sum(dim=(2, 3), keepdim=True))
        
    else:
        inv_placeholder = placeholder
    
    masked_img = new_images * masks + placeholder * (1 - masks)
    inverse_masked_img = new_images * inverse_masks + inv_placeholder * (1 - inverse_masks)
    return masked_img, inverse_masked_img


def create_square_mask(image_nrows: int, image_ncols: int, mask_percentage: float) -> th.Tensor:
    """Create a square mask of n_pixels in the image"""
    n_pixels = int(mask_percentage * image_nrows * image_ncols)
    square_nrows = int(n_pixels ** 0.5)
    mask = th.ones((image_nrows, image_ncols), dtype=th.float32)
    
    # Get a random top-left corner for the square
    row_idx = th.randint(0, image_ncols - square_nrows, (1,)).item()
    col_idx = th.randint(0, image_nrows - square_nrows, (1,)).item()
    
    mask[
        row_idx: row_idx + square_nrows,
        col_idx: col_idx + square_nrows
    ] = 0
    
    return mask