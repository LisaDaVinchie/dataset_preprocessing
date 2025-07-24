import torch as th
import torch.nn.functional as F
import random
import math
from typing import Tuple

mask_name = "masks"
square_mask_name = "square"
lines_mask_name = "lines"

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

def initialize_mask_kind(params: dict, mask_kind: str = None):
    """Initialize the mask kind based on the provided parameters."""
    mask_kind = params["dataset"]["mask_kind"] if mask_kind is None else mask_kind
    if mask_kind == square_mask_name:
        return SquareMask(params)
    elif mask_kind == lines_mask_name:
        return LinesMask(params)
    else:
        raise ValueError(f"Unknown mask kind: {mask_kind}")

class SquareMask:
    def __init__(self, params: dict = None, image_nrows: int = None, image_ncols: int = None, mask_percentage: float = None):
        """Initialize the SquareMask class.

        Args:
            params (dict, optional): the Json file containing the parameters. Defaults to None.
            image_nrows (int, optional): number of rows in the image to mask. Defaults to None.
            image_ncols (int, optional): number of columns in the image to mask. Defaults to None.
            mask_percentage (float, optional): fraction of pixels to mask, from 0 to 1. Defaults to None.
        """
        
        self.image_nrows = image_nrows
        self.image_ncols = image_ncols
        self.mask_percentage = mask_percentage

        if params is not None:
            self._initialize_parameters(params)

        self._check_parameters()
        
        n_pixels = int(self.mask_percentage * self.image_nrows * self.image_ncols)
        self.square_nrows = int(n_pixels ** 0.5)

    def _check_parameters(self):
        if self.image_nrows <= 0 or self.image_ncols <= 0:
            raise ValueError("Image dimensions must be positive integers.")
        
        if self.mask_percentage <= 0 or self.mask_percentage >= 1:
            raise ValueError("Mask percentage must be between 0 and 1.")

    def _initialize_parameters(self, params: dict):
        """Initialize parameters from a JSON file if provided. Priority is given to the parameters passed in the constructor.

        Args:
            params_path (dict): the JSON file containing parameters.

        Raises:
            ValueError: If any of the required parameters are None.
        """
        if params is not None:
            self.image_nrows = params['dataset']['cutted_nrows'] if self.image_nrows is None else self.image_nrows
            self.image_ncols = params['dataset']['cutted_ncols'] if self.image_ncols is None else self.image_ncols
            self.mask_percentage = params[mask_name][square_mask_name]['mask_percentage'] if self.mask_percentage is None else self.mask_percentage
        
        if self.image_nrows is None or self.image_ncols is None or self.mask_percentage is None:
            raise ValueError("Missing one of the following required parameters: image_nrows, image_ncols, mask_percentage")
        
    def mask(self) -> th.Tensor:
        """Create a square mask of n_pixels in the image

        Returns:
            th.Tensor: binary mask of shape (nrows, ncols), th.bool dtype, where False=masked, True=background
        """
        image_mask = th.ones((self.image_nrows, self.image_ncols), dtype=th.bool)
        
        # Get a random top-left corner for the square
        row_idx = th.randint(0, self.image_ncols - self.square_nrows, (1,)).item()
        col_idx = th.randint(0, self.image_nrows - self.square_nrows, (1,)).item()
        
        image_mask[
            row_idx: row_idx + self.square_nrows,
            col_idx: col_idx + self.square_nrows
        ] = False
        
        return image_mask

class LinesMask:
    def __init__(self, params: dict = None, image_nrows: int = None, image_ncols: int = None, num_lines: int = None, min_thickness: int = None, max_thickness: int = None):
        """Initialize the LinesMask class.

        Args:
            params (dict, optional): the Json file containing the parameters. Defaults to None.
            image_nrows (int, optional): number of rows in the image to mask. Defaults to None.
            image_ncols (int, optional): number of columns in the image to mask. Defaults to None.
            num_lines (int, optional): number of lines to generate. Defaults to None.
            min_thickness (int, optional): minimum line thickness. Defaults to None.
            max_thickness (int, optional): maximum line thickness. Defaults to None.
        """
        self.image_nrows = image_nrows
        self.image_ncols = image_ncols
        self.num_lines = num_lines
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        
        self._initialize_parameters(params)
        
        self._check_parameters()

    def _check_parameters(self):
        if self.image_nrows <= 0 or self.image_ncols <= 0:
            raise ValueError("Image dimensions must be positive integers.")
        
        if self.num_lines <= 0:
            raise ValueError("Number of lines must be a positive integer.")
        
        if self.min_thickness <= 0 or self.max_thickness < 0:
            raise ValueError("Line thickness must be positive integers.")
        
        if self.min_thickness > self.max_thickness:
            raise ValueError("Minimum thickness cannot be greater than maximum thickness.")
        

    def _initialize_parameters(self, params: dict):
        """Initialize parameters from a JSON file if provided. Priority is given to the parameters passed in the constructor.

        Args:
            params (dict): the JSON file containing parameters.

        Raises:
            ValueError: If any of the required parameters are None.
        """
        if params is not None:
                
            self.image_nrows = params['dataset']['cutted_nrows'] if self.image_nrows is None else self.image_nrows
            self.image_ncols = params['dataset']['cutted_ncols'] if self.image_ncols is None else self.image_ncols
            self.num_lines = params[mask_name][lines_mask_name]['num_lines'] if self.num_lines is None else self.num_lines
            self.min_thickness = params[mask_name][lines_mask_name]['min_thickness'] if self.min_thickness is None else self.min_thickness
            self.max_thickness = params[mask_name][lines_mask_name]['max_thickness'] if self.max_thickness is None else self.max_thickness
        
        if self.image_nrows is None or self.image_ncols is None or self.num_lines is None or self.min_thickness is None or self.max_thickness is None:
            raise ValueError("Missing one of the following required parameters: image_nrows, image_ncols, num_lines, min_thickness, max_thickness")
        
    def mask(self):
        """Create a mask of lines in the image

        Returns:
            th.Tensor: binary mask of shape (nrows, ncols), th.bool dtype, where False=masked, True=background
        """
        # Start with all ones (background)
        mask = th.ones((self.image_nrows, self.image_ncols), dtype=th.bool)
        
        for _ in range(self.num_lines):
            # Random start and end points
            start_point = (random.randint(0, self.image_nrows - 1), random.randint(0, self.image_ncols - 1))
            end_point = (random.randint(0, self.image_nrows - 1), random.randint(0, self.image_ncols - 1))
            
            # Random thickness
            thickness = random.randint(self.min_thickness, self.max_thickness)
            
            # Generate the line and subtract from mask (lines become False)
            mask = mask * (~self._generate_single_line(start_point, end_point, thickness))
            
        return mask

    def _generate_single_line(self, start_point: tuple, end_point: tuple, thickness: int) -> th.Tensor:
        """Helper function to generate a single line (1=line, 0=background).

        Args:
            start_point (tuple): start point of the line, as (row, col)
            end_point (tuple): end point of the line, as (row, col)
            thickness (int): thickness of the line, in pixels

        Returns:
            th.Tensor: binary mask of the line, of shape (nrows, ncols)
        """
        line_mask = th.zeros((self.image_nrows, self.image_ncols), dtype=th.bool)
        
        y1, x1 = start_point
        y2, x2 = end_point
        
        # Vector from start to end
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize
        length = max(math.sqrt(dx**2 + dy**2), 1e-8)
        dx /= length
        dy /= length
        
        # Generate points along the line
        num_samples = max(int(length * 2), 2)
        t_values = th.linspace(0, 1, num_samples)
        
        for t in t_values:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Create a grid for the thickness circle
            radius = thickness / 2
            i_values = th.arange(-thickness//2, thickness//2 + 1, dtype=th.float32)
            j_values = th.arange(-thickness//2, thickness//2 + 1, dtype=th.float32)
            
            for i in i_values:
                for j in j_values:
                    if (i**2 + j**2) <= radius**2:
                        yi = int(th.round(y + i).item())
                        xi = int(th.round(x + j).item())
                        if 0 <= yi < self.image_nrows and 0 <= xi < self.image_ncols:
                            line_mask[yi, xi] = True
                            
        return line_mask
    
class CloudMask:
    def __init__(self, params: dict = None, image_size: Tuple[int, int] = None):
        """Initialize the CloudMask class.

        Args:
            params (dict, optional): Parameters for cloud generation. Defaults to None.
            image_size (Tuple[int, int], optional): (height, width) of the image to mask. Defaults to None.
        """
        self.image_size = image_size
        self.coverage_range = (0.3, 0.9)
        self.default_params = {
            'base_threshold': 0.4,
            'res': 8,
            'octaves': 6,
            'persistence': 0.5
        }
        
        if params is not None:
            self.default_params.update(params)

    def mask(self, **kwargs) -> th.Tensor:
        """
        Create realistic boolean cloud mask resembling satellite imagery
        
        Args:
            base_threshold: base cutoff for cloud formation (0-1)
            res: base resolution parameter
            octaves: number of noise layers
            persistence: amplitude reduction per octave
            
        Returns:
            th.Tensor: Boolean mask where True indicates cloud coverage
        """
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Generate continuous cloud probability mask
        cloud_prob = self._generate_cloud_probability_mask(self.image_size, **params)
        
        # Convert to boolean mask
        coverage = random.uniform(*self.coverage_range)
        bool_mask = self._threshold_to_coverage(cloud_prob, coverage)
        
        return ~bool_mask

    def _generate_cloud_probability_mask(self, shape: Tuple[int, int], 
                                      base_threshold: float,
                                      res: int,
                                      octaves: int,
                                      persistence: float) -> th.Tensor:
        """Generate continuous cloud probability mask"""
        # Generate fractal noise
        noise = self._generate_fractal_noise(shape, res, octaves, persistence)
        
        # Apply base threshold with variability
        threshold = base_threshold + 0.1 * th.randn(1).item()
        mask = th.sigmoid((noise - threshold) * 10)  # Soft threshold
        
        # Apply smoothing
        mask = self._smooth_mask(mask)
        
        # Add texture variability
        texture = self._generate_fractal_noise(shape, res*2, octaves-2, persistence*1.2)
        mask = th.clamp(mask * (0.8 + 0.4 * texture), 0, 1)
        
        # Add atmospheric effects
        mask = self._add_atmospheric_effects(mask)
        
        return mask

    def _generate_fractal_noise(self, shape: Tuple[int, int], 
                              res: int, 
                              octaves: int, 
                              persistence: float) -> th.Tensor:
        """Generate fractal noise for natural cloud patterns"""
        noise = th.zeros(shape)
        frequency = 1
        amplitude = 1
        
        for _ in range(octaves):
            # Generate scaled random noise
            h, w = shape
            scaled_shape = (max(1, int(h/res/frequency)), max(1, int(w/res/frequency)))
            rand_noise = th.randn(scaled_shape)
            
            # Upscale with bilinear interpolation
            upscaled = F.interpolate(rand_noise.unsqueeze(0).unsqueeze(0), 
                                    size=shape, mode='bilinear', align_corners=False).squeeze()
            
            noise += upscaled * amplitude
            frequency *= 2
            amplitude *= persistence
        
        # Normalize to 0-1 range
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        return noise

    def _threshold_to_coverage(self, prob_mask: th.Tensor, 
                             target_coverage: float) -> th.Tensor:
        """Convert probability mask to boolean mask with exact coverage"""
        sorted_values = th.sort(prob_mask.flatten())[0]
        target_pixels = int(target_coverage * prob_mask.numel())
        threshold = sorted_values[-target_pixels]
        return prob_mask >= threshold

    def _smooth_mask(self, mask: th.Tensor, kernel_size: int = 7) -> th.Tensor:
        """Apply Gaussian smoothing to mask"""
        kernel = self._get_gaussian_kernel(kernel_size)
        return F.conv2d(mask.unsqueeze(0).unsqueeze(0), 
                       kernel, 
                       padding=kernel_size//2).squeeze()

    def _get_gaussian_kernel(self, size: int, sigma: float = 1.0) -> th.Tensor:
        """Create Gaussian kernel for smoothing"""
        coords = th.arange(size).float() - size//2
        g = th.exp(-(coords**2) / (2 * sigma**2))
        g = g.ger(g)  # Outer product
        return (g / g.sum()).view(1, 1, size, size)

    def _add_atmospheric_effects(self, mask: th.Tensor) -> th.Tensor:
        """Add subtle atmospheric effects for realism"""
        # Edge detection
        edges = F.avg_pool2d(mask.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2) - \
                F.avg_pool2d(1 - mask.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2)
        edges = edges.squeeze()
        
        # Apply edge effects
        edge_region = (edges > -0.2) & (edges < 0.2)
        mask[edge_region] = th.clamp(mask[edge_region] + 0.2 * th.randn_like(mask[edge_region]), 0, 1)
        
        return mask