import torch as th
from pathlib import Path
from time import time
from datetime import datetime
import random
import json
import os
import sys
import math
import pickle
from typing import Type

path_to_append = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_to_append)
from utils.mask_data import initialize_mask_kind
from utils.dataset_normalization import MinMaxNormalization
from utils.parse_params import parse_input_paths

def main():
    start_time = time()
    params, paths = parse_input_paths()
        
    n_cutted_images = int(params["dataset"]["n_cutted_images"])
    same_mask = str(params["dataset"]["same_mask"]).lower() == "true"
    nan_placeholder = params["dataset"]["nan_placeholder"]
    if nan_placeholder == "false":
        raise ValueError("The placeholder value must be a float, not 'false'.")
    
    dataset_kind = str(params["dataset"]["dataset_kind"])
    n_channels = int(params["dataset"][dataset_kind]["n_channels"])
    masked_channels = list(params["dataset"][dataset_kind]["masked_channels"])
        
    print("Using placeholder value: ", nan_placeholder, flush=True)
    
    processed_data_dir = Path(paths["data"]["processed_data_dir"][dataset_kind])
    next_minimal_dataset_path = Path(paths["data"]["next_minimal_dataset_path"])
    dataset_specs_path = Path(paths["data"]["dataset_specs_path"])
    next_nans_masks_path = Path(paths["data"]["next_nans_masks_path"])
    next_minmax_path = Path(paths["data"]["next_minmax_path"])
    
    # Check if the directories exist
    check_dirs_existance([processed_data_dir, next_minimal_dataset_path.parent, dataset_specs_path.parent, next_minmax_path.parent])

    # Select n_images random images from the processed images
    processed_images_paths = list(processed_data_dir.glob(f"*.pt"))

    if len(processed_images_paths) == 0:
        raise FileNotFoundError(f"No images found in {processed_data_dir}")

    print(f"\nFound {len(processed_images_paths)} images in {processed_data_dir}\n", flush=True)
    
    cut_class = CutAndMaskImage(params=params)

    # Select some random points, to use as centers for the cutted images
    idx_time = time()
    random_points = cut_class.select_random_points(n_points=n_cutted_images)
    print(f"Selected random points for cutted images in {time() - idx_time} seconds\n", flush=True)

    path_to_indices = cut_class.map_random_points_to_images(processed_images_paths, random_points)
    
    print(f"Mapped the points to images in {time() - idx_time} seconds\n", flush=True)

    d_time = time() 
    # Generate the dataset
    dataset, nans_masks = cut_class.generate_image_dataset(n_channels=n_channels,
                                                           masked_channels_list=masked_channels, path_to_indices_map=path_to_indices,
                                                           placeholder=nan_placeholder, same_mask=same_mask)
    print(f"Generated the dataset in {time() - d_time} seconds\n", flush=True)
    
    norm_class = MinMaxNormalization(batch_size=1000)

    if dataset is not None:
        dataset["images"], minmax = norm_class.normalize(dataset["images"], nans_masks)
        
        pickle.HIGHEST_PROTOCOL = 4
        th.save(dataset, next_minimal_dataset_path, _use_new_zipfile_serialization=False)
        print(f"Saved the minimal dataset to {next_minimal_dataset_path}\n", flush=True)
        th.save(minmax, next_minmax_path)
        print(f"Saved the minmax values to {next_minmax_path}\n", flush=True)
        
        th.save(nans_masks, next_nans_masks_path)
        print(f"Saved the nans masks to {next_nans_masks_path}\n", flush=True)

    # Extract the "dataset" and "mask" sections
    dataset_section = params["dataset"]
    masks = params["masks"]

    # Combine the sections into a single dictionary
    sections_to_save = {
        'dataset': dataset_section,
        'masks': masks
    }
    
    # Save the combined sections to a text file
    with open(dataset_specs_path, 'w') as f:
        json.dump(sections_to_save, f, indent=4)

    print("Elapsed time: {:.2f} seconds\n".format(time() - start_time), flush=True)
class CutAndMaskImage:
    """Class used to:
        - select random smaller images from a larger image
        - mask some pixels in the smaller images to simulate missing data
        - adding a time layer to the images, indicating the date
        - save the dataset and masks
        - save info regarding the dataset, such as the number of images, the size of the images, etc.
    """
    def __init__(self, params: dict = None, original_nrows: int = None, original_ncols: int  = None, final_nrows: int  = None, final_ncols: int  = None, nans_threshold: float  = None, n_cutted_images: int  = None, mask_kind: str = None, mask_function: Type = None):
        """Initalize the class with the parameters needed to cut the images and generate the dataset

        Args:
            params (dict): parameters to use to initialize the class. If None, the parameters will be loaded from the params file
            original_nrows (int): number of rows in the original image
            original_ncols (int): number of columns in the original image
            final_nrows (int): number of rows in the cutted image
            final_ncols (int): number of columns in the cutted image
            nans_threshold (float): maximum fraction of nans allowed in the cutted image. Must be between 0 and 1
            n_cutted_images (int): number of cutted images to generate
            mask_kind (str): kind of mask to use.
            mask_function (Type): function to use to generate the mask. If None, the function will be initialized using the mask_kind parameter
        """
        self.original_nrows = original_nrows
        self.original_ncols = original_ncols
        self.cutted_nrows = final_nrows
        self.cutted_ncols = final_ncols
        self.nans_threshold = nans_threshold
        self.n_cutted_images = n_cutted_images
        self.mask_kind = mask_kind
        self.mask_function = mask_function
        
        self._load_parameters(params)
        
        self.mask_function = initialize_mask_kind(params, self.mask_kind)
        
        self._check_params()

    def _load_parameters(self, params):
        if params is not None:
            
            dataset_kind = str(params["dataset"]["dataset_kind"])
            self.original_nrows = int(params["dataset"][dataset_kind]["x_shape_raw"]) if self.original_nrows is None else self.original_nrows
            self.original_ncols = int(params["dataset"][dataset_kind]["y_shape_raw"]) if self.original_ncols is None else self.original_ncols
            self.n_cutted_images = int(params["dataset"]["n_cutted_images"]) if self.n_cutted_images is None else self.n_cutted_images
            self.cutted_nrows = int(params["dataset"]["cutted_nrows"]) if self.cutted_nrows is None else self.cutted_nrows
            self.cutted_ncols = int(params["dataset"]["cutted_ncols"]) if self.cutted_ncols is None else self.cutted_ncols
            self.nans_threshold = float(params["dataset"]["nans_threshold"]) if self.nans_threshold is None else self.nans_threshold
            if self.mask_function is None:
                self.mask_kind = str(params["dataset"]["mask_kind"]) if self.mask_kind is None else None
                
        if self.mask_function is None and self.mask_kind is None:
            raise ValueError("The mask function and the mask kind are both None. Please provide one of them.")
        
        for param in [self.original_nrows, self.original_ncols, self.cutted_nrows, self.cutted_ncols, self.nans_threshold, self.n_cutted_images]:
            if param is None:
                raise ValueError("Some parameters are None. Please check the input parameters.")

    def _check_params(self):
        if self.original_nrows <=0 or self.original_ncols <= 0:
            raise ValueError("The original image dimensions must be greater than 0.")
        if self.cutted_nrows <=0 or self.cutted_ncols <= 0:
            raise ValueError("The cutted image dimensions must be greater than 0.")
        if self.nans_threshold < 0 or self.nans_threshold > 1:
            raise ValueError("The nans threshold must be between 0 and 1.")
        if self.n_cutted_images <= 0:
            raise ValueError("The number of cutted images must be greater than 0.")
        if self.cutted_nrows > self.original_nrows or self.cutted_ncols > self.original_ncols:
            raise ValueError("The cutted image dimensions must be smaller than the original image dimensions.")
        if self.mask_function is not None and not hasattr(self.mask_function, 'mask'):
            raise AttributeError("The provided mask_function does not have a 'mask' method.")

    def select_random_points(self, n_points: int) -> th.Tensor:
        """Select random points in the original image, to use as top-left corners for the cutted images

        Args:
            n_points (int): number of random points to select

        Returns:
            th.Tensor: tensor with the selected random points, as (x, y) coordinates
        """
        
        random_x = th.randint(0, self.original_nrows - self.cutted_nrows, (n_points,))
        random_y = th.randint(0, self.original_ncols - self.cutted_ncols, (n_points,))
        random_points = th.stack([random_x, random_y], dim = 1)
        return random_points

    def map_random_points_to_images(self, image_file_paths: list, selected_random_points: th.Tensor) -> dict:
        """Assign the points to some random images

        Args:
            image_file_paths (list): paths to the images
            selected_random_points (th.Tensor): points from a tensor

        Returns:
            dict: dictionary with the paths to the images as keys and the points as values
        """
        
        # Chose one image for each point. Images can be used multiple times
        chosen_paths = [random.choice(image_file_paths) for _ in range(len(selected_random_points))]
        path_to_indices = {}
        
        for path, point in zip(chosen_paths, selected_random_points):
            if path not in path_to_indices:
                path_to_indices[path] = []
        
            path_to_indices[path].append(point)
        return path_to_indices
    
    def cut_valid_image(self, image: th.Tensor, index: list, n_pixel_threshold: int) -> th.Tensor:
        """Cut a valid image from the original image, checking that the cutted image does not contain too many nans.

        Args:
            image (th.Tensor): original image
            index (int): position of the top-left corner of the cutted image in the original image
            n_pixel_threshold (int): maximum number of nans allowed in the cutted image. If the number of nans is greater than this threshold, the function will select a new random point

        Returns:
            th.Tensor: cutted image, of shape (n_channels, final_nrows, final_ncols)
        """
        
        cutted_img = image[:, index[0]:index[0] + self.cutted_nrows, index[1]:index[1] + self.cutted_ncols]
       
        nan_count = th.isnan(cutted_img).sum().item()
        if nan_count > n_pixel_threshold:
            while nan_count > n_pixel_threshold:
                index = self.select_random_points(1)[0]
                
                cutted_img = image[:, index[0]:index[0] + self.cutted_nrows, index[1]:index[1] + self.cutted_ncols]
                nan_count = th.isnan(cutted_img).sum().item()
        
        return cutted_img
    
    # def batch_cut_images(self, image: th.Tensor, indices: list, threshold: int) -> th.Tensor:
    #     """Vectorized version of cut_valid_image for multiple indices"""
    #     print(indices)
    #     indices_tensor = th.cat([th.tensor(idx).unsqueeze(0) for idx in indices], dim=0)  # Flatten into a single tensor
    #     windows = indices_tensor.unsqueeze(1).expand(-1, 2, -1)  # Shape: [n_indices, 2, 2]
    #     windows[..., 0] += self.cutted_nrows  # Add row offsets
    #     windows[..., 1] += self.cutted_ncols   # Add col offsets
        
    #     # Batched slicing
    #     cuts = image[:, windows[:, 0, 0]:windows[:, 0, 1], 
    #                 windows[:, 1, 0]:windows[:, 1, 1]]
        
    #     # Vectorized NaN check
    #     nan_counts = th.isnan(cuts).sum(dim=(1,2,3))
    #     valid = nan_counts <= threshold
        
    #     # Only re-cut invalid slices
    #     while not valid.all():
    #         bad_idx = ~valid
    #         new_indices = self.select_random_points(bad_idx.sum().item())
    #         new_cuts = self.batch_cut_images(image, new_indices, threshold)
    #         cuts[bad_idx] = new_cuts[bad_idx]
    #         valid[bad_idx] = True
            
    #     return cuts

    def generate_image_dataset(self, n_channels: int, masked_channels_list: list, path_to_indices_map: dict, placeholder: float, same_mask: bool = False) -> tuple[dict, th.Tensor]:
        """Generate a dataset of masked images, inverse masked images and masks

        Args:
            n_channels (int): final number of channels in the image
            masked_channels_list (list): list of channels that should not be masked
            path_to_indices_map (dict): dictionary with the paths to the images as keys and the points as values
            placeholder (float): value to use as placeholder for nan pixels.
            same_mask (bool): if True, use the same mask for all the masked channels of the image, otherwise use a different mask for each channel. Defaults to False.

        Returns:
            tuple: dictionary with the images and masks and the nans masks
        """
        
        # Initialize the two datasets
        dataset = None
        
        dataset_keys = ["images", "masks"]
        dataset_shape = (self.n_cutted_images, n_channels, self.cutted_nrows, self.cutted_ncols)
        dataset = {dataset_keys[0]: th.empty(dataset_shape, dtype=th.float32),
                   dataset_keys[1]: th.empty(dataset_shape, dtype=th.bool)}
            
        nans_masks = th.ones((self.n_cutted_images, n_channels, self.cutted_nrows, self.cutted_ncols), dtype=th.bool)
        
        masked_channels_list = list(masked_channels_list)
        
        # Calculate the interval between the oldest and newest date, in days
        selected_dates = [Path(path).stem for path in path_to_indices_map.keys()]
        dates_parsed = [datetime.strptime(d, "%Y_%m_%d").date() for d in selected_dates]
        oldest_date, newest_date = min(dates_parsed), max(dates_parsed)
        interval = (newest_date - oldest_date).days

        idx_start = 0 # index of the first cutted image of this raw image
        idx_end = 0 # index of the last cutted image of this raw image
        n_original_channels = n_channels - 1 # The last channel is the time layer
        n_pixels = self.cutted_nrows * self.cutted_ncols * n_original_channels # number of pixels in the raw image
        threshold = self.nans_threshold * n_pixels # threshold of nans in the image
        
        image_cache = {}
        for path, indices in path_to_indices_map.items():
            
            if path not in image_cache:
                image = th.load(path)
                image_cache[path] = image
            else:
                image = image_cache[path]
            n_indices = len(indices)
            idx_end = idx_start + n_indices
            
            # Add the time layer to the images
            days_from_inital_date = (datetime.strptime(Path(path).stem, "%Y_%m_%d").date() - oldest_date).days
            encoded_time = math.cos(math.pi * days_from_inital_date / interval)
            
            # Generate the cutted images, adding the time layer
            # cutted_imgs = self.batch_cut_images(image, indices, threshold)
            
            with th.no_grad():
                cutted_imgs = th.stack([self.cut_valid_image(image, index, threshold) for index in indices], dim=0)
            
            time_layers = th.ones((n_indices, 1, self.cutted_nrows, self.cutted_ncols), dtype=th.float32) * encoded_time
            cutted_imgs = th.cat((cutted_imgs, time_layers), dim=1)
            
            # Find where the nans are in the cutted images
            cutted_img_nans = ~th.isnan(cutted_imgs)
            nans_masks[idx_start:idx_end, :, :, :] = cutted_img_nans
            
            # Create square masks. 0 where the values are masked, 1 where the values are not masked
            masks = th.ones((n_indices, n_channels, self.cutted_nrows, self.cutted_ncols), dtype=th.bool)
            
            if same_mask:
                image_mask = self.mask_function.mask()
                masks[:, masked_channels_list, :, :] = image_mask
            else:
                for mc in masked_channels_list:
                    # Generate the mask for each channel
                    masks[:, mc, :, :] = self.mask_function.mask()
                # all_masks = th.stack([self.mask_function.mask() for _ in masked_channels_list])
            # Set masks to 0 where the nan mask is 0
            masks = th.where(cutted_img_nans == False, th.tensor(False, dtype=masks.dtype), masks)
            
            # Save the images to the minimal dataset, substituting the nans with the placeholder
            dataset[dataset_keys[0]][idx_start:idx_end] = th.nan_to_num(cutted_imgs, nan=placeholder)
            dataset[dataset_keys[1]][idx_start:idx_end] = masks
            
            idx_start = idx_end
            
        return dataset, nans_masks

def check_dirs_existance(dirs: list[Path]):
    """Check if the directories exist

    Args:
        dirs (list[Path]): list of directories to check

    Raises:
        FileNotFoundError: if a directory does not exist
    """
    for dir in dirs:
        if not dir.exists():
            raise FileNotFoundError(f"Folder {dir} does not exist.")
    
if __name__ == "__main__":
    main()