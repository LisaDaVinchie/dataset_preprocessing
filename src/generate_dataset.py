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
import calendar

path_to_append = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_to_append)
from utils.mask_data import initialize_mask_kind
from utils.dataset_normalization import MinMaxNormalization
from utils.parse_params import parse_input_paths

def main():
    start_time = time()
    params, paths = parse_input_paths()
        
    n_cutted_images = int(params["dataset"]["n_cutted_images"])
    mask_kind = str(params["dataset"]["mask_kind"])
    same_mask = str(params["dataset"]["same_mask"]).lower() == "true"
    nan_placeholder = params["dataset"]["nan_placeholder"]
    if nan_placeholder == "false":
        raise ValueError("The placeholder value must be a float, not 'false'.")
    
    dataset_kind = str(params["dataset"]["dataset_name"])
    masked_channels = list(params[dataset_kind]["masked_channels"])
    channels_to_keep = list(params[dataset_kind]["channels_to_keep"])
    n_channels = len(channels_to_keep) + 1 # +1 for the time layer
        
    print("Using placeholder value: ", nan_placeholder, flush=True)
    
    processed_data_dir = Path(paths[dataset_kind]["processed_data_dir"])
    dataset_path = Path(paths["dataset"]["next_dataset_path"])
    specs_path = Path(paths["dataset"]["next_specs_path"])
    nanmasks_path = Path(paths["dataset"]["next_nanmasks_path"])
    minmax_path = Path(paths["dataset"]["next_minmax_path"])
    
    # Check if the directories exist
    check_dirs_existance([processed_data_dir, dataset_path.parent, specs_path.parent, minmax_path.parent])

    # Select n_images random images from the processed images
    processed_images_paths = list(processed_data_dir.glob(f"*.pt"))

    if len(processed_images_paths) == 0:
        raise FileNotFoundError(f"No images found in {processed_data_dir}")

    print(f"\nFound {len(processed_images_paths)} images in {processed_data_dir}\n", flush=True)
    
    cut_class = CutImages(params=params)

    # Select some random points, to use as centers for the cutted images
    idx_time = time()
    random_points = cut_class.select_random_points(n_points=n_cutted_images)
    print(f"Selected random points for cutted images in {time() - idx_time} seconds\n", flush=True)

    available_days = cut_class._get_available_days(processed_images_paths)
    path_to_indices = cut_class.map_random_points_to_days(available_days, random_points)
    
    print(f"Mapped the points to images in {time() - idx_time} seconds\n", flush=True)

    d_time = time()
    # Generate the dataset
    
    images, nans_masks = cut_class.generate_cutted_images(n_channels=n_channels, path_to_indices_map=path_to_indices, placeholder=nan_placeholder)
    print(f"Generated the dataset in {time() - d_time} seconds\n", flush=True)
    print(f"Number of nans in the dataset before normalization: {th.isnan(images).sum()}", flush=True)
    
    norm_class = MinMaxNormalization(batch_size=1000)

    dataset = {}
    dataset["images"], minmax = norm_class.normalize(images, nans_masks)
    
    print(f"Number of nans in the dataset: {th.isnan(dataset['images']).sum()}", flush=True)
    
    dataset["masks"] = th.ones_like(images, dtype=th.float32)
    mask_class = initialize_mask_kind(params, mask_kind)
    for j in masked_channels:
        dataset["masks"][:, j, :, :] = th.stack([mask_class.mask() for _ in range(images.shape[0])], dim=0)
    
        
    pickle.HIGHEST_PROTOCOL = 4
    th.save(dataset, dataset_path, _use_new_zipfile_serialization=False)
    print(f"Saved the minimal dataset to {dataset_path}\n", flush=True)
    th.save(minmax, minmax_path)
    print(f"Saved the minmax values to {minmax_path}\n", flush=True)
    
    th.save(nans_masks, nanmasks_path)
    print(f"Saved the nans masks to {nanmasks_path}\n", flush=True)

    # Extract the "dataset" and "mask" sections
    dataset_section = params["dataset"]
    masks = params["masks"]

    # Combine the sections into a single dictionary
    sections_to_save = {
        'dataset': dataset_section,
        'masks': masks
    }
    
    # Save the combined sections to a text file
    with open(specs_path, 'w') as f:
        json.dump(sections_to_save, f, indent=4)

    print("Elapsed time: {:.2f} seconds\n".format(time() - start_time), flush=True)
    
class CutImages:
    """Class used to:
        - select random smaller images from a larger image
        - mask some pixels in the smaller images to simulate missing data
        - adding a time layer to the images, indicating the date
        - save the dataset and masks
        - save info regarding the dataset, such as the number of images, the size of the images, etc.
    """
    def __init__(self, params: dict = None, original_nrows: int = None, original_ncols: int  = None, final_nrows: int  = None, final_ncols: int  = None, nans_threshold: float  = None, n_cutted_images: int  = None):
        """Initalize the class with the parameters needed to cut the images and generate the dataset

        Args:
            params (dict): parameters to use to initialize the class. If None, the parameters will be loaded from the params file
            original_nrows (int): number of rows in the original image
            original_ncols (int): number of columns in the original image
            final_nrows (int): number of rows in the cutted image
            final_ncols (int): number of columns in the cutted image
            nans_threshold (float): maximum fraction of nans allowed in the cutted image. Must be between 0 and 1
            n_cutted_images (int): number of cutted images to generate
        """
        self.original_nrows = original_nrows
        self.original_ncols = original_ncols
        self.cutted_nrows = final_nrows
        self.cutted_ncols = final_ncols
        self.nans_threshold = nans_threshold
        self.n_cutted_images = n_cutted_images
        
        self._load_parameters(params)
        
        self._check_params()

    def _load_parameters(self, params):
        if params is not None:
            
            dataset_kind = str(params["dataset"]["dataset_name"])
            dataset_params = params[dataset_kind]
            self.original_nrows = int(dataset_params["n_rows"]) if self.original_nrows is None else self.original_nrows
            self.original_ncols = int(dataset_params["n_cols"]) if self.original_ncols is None else self.original_ncols
            self.n_cutted_images = int(params["dataset"]["n_cutted_images"]) if self.n_cutted_images is None else self.n_cutted_images
            self.cutted_nrows = int(params["dataset"]["cutted_nrows"]) if self.cutted_nrows is None else self.cutted_nrows
            self.cutted_ncols = int(params["dataset"]["cutted_ncols"]) if self.cutted_ncols is None else self.cutted_ncols
            self.nans_threshold = float(params["dataset"]["nans_threshold"]) if self.nans_threshold is None else self.nans_threshold
        
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

    def _get_available_days(self, path_list: list) -> dict:
        """Get the available days in the month list

        Args:
            path_list (list): list of paths in the format path_to_folder/YYYY_MM.pt

        Returns:
            dict: dictionary with the available days in the month list
        """
        
        available_days = {}

        for path in path_list:
            year, month = map(int, Path(path).stem.split("_"))
            num_days = calendar.monthrange(year, month)[1]
            available_days[path] = [day for day in range(1, num_days + 1)]
        return available_days
        
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

    def map_random_points_to_days(self, days_list: dict, points_list: list) -> dict:
        """Assign the points to some random images

        Args:
            days_list (dict): dictionary with the available days in the month list
            points_list (th.Tensor): points from a tensor

        Returns:
            dict: dictionary with the months as keys and the tuples (day, point) as values
        """
        
        paths = list(days_list.keys())
        
        points_to_days = {}
        
        for point in points_list:
            path = random.choice(paths)
            day = random.choice(days_list[path])
            if path not in points_to_days:
                points_to_days[path] = []
            points_to_days[path].append((day, point))
        
        return points_to_days
    
    def _cut_valid_image(self, image: th.Tensor, index: list, n_pixel_threshold: int) -> th.Tensor:
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

    def _calculate_time_interval(self, path_to_indices_map: dict) -> tuple:
        """Calculate the interval between the oldest and newest date, in days

        Args:
            path_to_indices_map (dict): dictionary with the paths to the images as keys and the points as values

        Returns:
            tuple: oldest date, newest date, interval in days
        """
        
        selected_dates = []
        for path in path_to_indices_map.keys():
            month = Path(path).stem
            days = [d[0] for d in path_to_indices_map[path]]
            selected_dates.extend([f"{month}_{day:02d}" for day in days])
        
        dates_parsed = [datetime.strptime(d, "%Y_%m_%d").date() for d in selected_dates]
        oldest_date, newest_date = min(dates_parsed), max(dates_parsed)
        interval = (newest_date - oldest_date).days
        
        return oldest_date, newest_date, interval
    
    def _get_encoded_time(self, date: str, oldest_date: datetime.date, interval: int) -> float:
        days_from_inital_date = (datetime.strptime(date, "%Y_%m_%d").date() - oldest_date).days
        return math.cos(math.pi * days_from_inital_date / interval)
    
    def generate_cutted_images(self, n_channels: int, path_to_indices_map: dict, placeholder: float) -> tuple[th.Tensor, th.Tensor]:
        """Generate a dataset of masked images, inverse masked images and masks

        Args:
            n_channels (int): final number of channels in the image
            path_to_indices_map (dict): dictionary with the paths to the images as keys and the points as values
            placeholder (float): value to use as placeholder for nan pixels.

        Returns:
            tuple: tuple with the cutted images and the masks
        """
        
        # Initialize the datasets
        images_shape = (self.n_cutted_images, n_channels, self.cutted_nrows, self.cutted_ncols)
        dataset = th.empty(images_shape, dtype=th.float32)
            
        nans_masks = th.ones((self.n_cutted_images, n_channels, self.cutted_nrows, self.cutted_ncols), dtype=th.bool)
        
        oldest_date, _, interval = self._calculate_time_interval(path_to_indices_map)

        idx_start = 0 # index of the first cutted image of this raw image
        idx_end = 0 # index of the last cutted image of this raw image
        n_original_channels = n_channels - 1 # The last channel is the time layer
        n_pixels = self.cutted_nrows * self.cutted_ncols * n_original_channels # number of pixels in the raw image
        threshold = self.nans_threshold * n_pixels # threshold of nans in the image
        
        # Loop over the images
        for path in sorted(path_to_indices_map.keys()):
            raw_data = th.load(path)
            print(f"Processing {path}...", flush=True)
            day_index_list = path_to_indices_map[path]
            n_days = len(day_index_list) # number of (day, point) pairs in the image
            idx_end = idx_start + n_days
            encoded_times = [self._get_encoded_time(f"{Path(path).stem}_{day:02d}", oldest_date, interval) for (day, _) in day_index_list]
            time_layers = th.stack([th.ones((self.cutted_nrows, self.cutted_ncols), dtype=th.float32) * encoded_time for encoded_time in encoded_times], dim=0)
            dataset[idx_start:idx_end, -1, :, :] = time_layers
            
            cutted_images_list = [self._cut_valid_image(raw_data[day - 1, :, :, :], point, threshold) for (day, point) in day_index_list]
            temp_imgs = th.stack(cutted_images_list, dim=0)
            nans_masks[idx_start:idx_end, :-1, :, :] = ~th.isnan(temp_imgs)
            dataset[idx_start:idx_end, :-1, :, :] = th.nan_to_num(temp_imgs, nan=placeholder)
            idx_start = idx_end
            print(f"Processed file {path}\n")
                
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