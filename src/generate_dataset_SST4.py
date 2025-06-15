########################################################################
#
#   Generate non normalized dataset
#   DINCAE 2.0 style
#
#########################################################################


import torch as th
from pathlib import Path
import math
import random
import calendar
from datetime import datetime, timedelta
from time import time
import json
import os
import sys

path_to_append = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_to_append)
from utils.mask_data import initialize_mask_kind
from utils.parse_params import parse_input_paths

def main():
    start_time = time()
    params, paths = parse_input_paths()
    
    dataset_kind = str(params["dataset"]["dataset_kind"])
    nan_placeholder = params["dataset"]["nan_placeholder"]
    
    if nan_placeholder is None:
        raise ValueError("The nan placeholder must be set in the parameters.")
    
    # Paths to the input data
    processed_data_dir = Path(paths[dataset_kind]["processed_data_dir"])
    original_nan_masks_dir = Path(paths[dataset_kind]["nan_masks_dir"])
    
    if not processed_data_dir.exists():
        raise ValueError(f"Processed data directory {processed_data_dir} does not exist.")
    if not original_nan_masks_dir.exists():
        raise ValueError(f"Original nan masks directory {original_nan_masks_dir} does not exist.")
    
    # Paths to the outputs
    dataset_paths = paths["dataset"]
    dataset_path = Path(dataset_paths["next_dataset_path"])
    specs_path = Path(dataset_paths["next_specs_path"])
    nanmasks_path = Path(dataset_paths["next_nanmasks_path"])
    minmax_path = Path(dataset_paths["next_minmax_path"])
    for path in [dataset_path, specs_path, nanmasks_path, minmax_path]:
        if not path.parent.exists():
            raise ValueError(f"Output directory {path.parent} does not exist.")
    
    # Initialize the mask kind
    cut = CutImages(params)
    print("Class initialized\n")
    
    # Get the paths to the files
    original_nan_masks_paths = list(original_nan_masks_dir.glob("*.pt"))
    if len(original_nan_masks_paths) == 0:
        raise ValueError(f"No files found in {original_nan_masks_dir}.")
    original_nan_masks_paths.sort()
    
    # Map the random points to the days
    
    print("Mapping the random points to the days")
    final_dict, _, masks = cut.get_data_paths_to_dataset_idx_dict(original_nan_masks_paths, processed_data_dir)
    print("Mapping done\n")
    
    # Get the paths to the files
    original_data_paths = list(processed_data_dir.glob("*.pt"))
    if len(original_data_paths) == 0:
        raise ValueError(f"No files found in {processed_data_dir}.")
    original_data_paths.sort()
    
    # Cut the images
    print("Cutting the images")
    cutted_images = cut.cut(final_dict, original_data_paths)
    print("Cutting done\n")
    nan_masks = ~th.isnan(cutted_images)
    print("Nan masks created\n")
    
    print("Creating dataset")
    dataset = {}
    norm_images = th.nan_to_num(cutted_images, nan=nan_placeholder)
    dataset["images"] = norm_images
    dataset["masks"] = th.logical_and(nan_masks, masks)
    
    # dataset["masks"] = create_masks(params, cutted_images, nan_masks)
    
    print("Dataset created\n")
    
    # Save the cutted images
    print("Saving dataset")
    th.save(dataset, dataset_path, _use_new_zipfile_serialization=False)
    print(f"Dataset saved to {dataset_path}\n")
    
    # Save the nan masks
    print("Saving nan masks")
    th.save(nan_masks, nanmasks_path, _use_new_zipfile_serialization=False)
    print(f"Nan masks saved to {nanmasks_path}\n")
    
    print("Saving specs")
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

    print(f"Specs saved to {specs_path}\n")
    
    print("Elapsed time: ", time() - start_time)

def create_masks(params, cutted_images, nan_masks):
    masks = th.ones_like(cutted_images, dtype=th.bool)
    
    total_days = int(params["dataset"]["total_days"])
    
    c = (total_days // 2)
    
    mask_class = initialize_mask_kind(params)
    n_images = cutted_images.shape[0]
    
    mask_w = mask_class.square_nrows
    
    max_pixels = int(0.3 * mask_w * mask_w)
    max_trials = 1000
    
    for i in range(n_images):
        mask = mask_class.mask()
        n_nans_in_mask = th.sum(~nan_masks[i, c, :, :] & ~mask)
        trials = 0
        while n_nans_in_mask > max_pixels and trials < max_trials:
            mask = mask_class.mask()
            n_nans_in_mask = th.sum(~nan_masks[i, c, :, :] & ~mask)
            trials += 1
        if trials == max_trials:
            raise ValueError(f"Could not create a valid mask for image {i} after {max_trials} trials. Please check the parameters.")
    
    return th.logical_and(masks, nan_masks)
    
class CutImages:
    def __init__(self, params, original_nrows: int = None, original_ncols: int = None, final_nrows: int = None, final_ncols: int = None, n_images: int = None, nans_perc: float = None, total_days: int = None, max_trials: int = 200):
        self.params = params
        self.original_nrows = original_nrows
        self.original_ncols = original_ncols
        self.final_nrows = final_nrows
        self.final_ncols = final_ncols
        self.n_images = n_images
        self.nans_threshold = nans_perc
        self.max_trials = max_trials
        self.total_days = total_days
        self._load_parameters(params)
        self._check_params()
        self.surrounding_days = self.total_days // 2
        self.max_pixels = int(self.final_nrows * self.final_ncols * self.nans_threshold)
        
        self.mask_class = initialize_mask_kind(params)
        
    def _load_parameters(self, params):
        if params is not None:
            dataset_params = params["dataset"]
            dataset_kind = str(dataset_params["dataset_kind"])
            if self.original_nrows is None:
                self.original_nrows = int(dataset_params[dataset_kind]["n_rows"])
            if self.original_ncols is None:
                self.original_ncols = int(dataset_params[dataset_kind]["n_cols"])
            if self.final_nrows is None:
                self.final_nrows = int(dataset_params["cutted_nrows"])
            if self.final_ncols is None:
                self.final_ncols = int(dataset_params["cutted_ncols"])
            if self.n_images is None:
                self.n_images = int(dataset_params["n_cutted_images"])
            if self.nans_threshold is None:
                self.nans_threshold = float(dataset_params["nans_threshold"])
            if self.total_days is None:
                self.total_days = int(dataset_params["total_days"])
        
        for param in [self.original_nrows, self.original_ncols, self.final_nrows, self.final_ncols, self.nans_threshold, self.n_images, self.total_days]:
            if param is None:
                raise ValueError("Some parameters are None. Please check the input parameters.")
            
    def _check_params(self):
        if self.original_nrows <=0 or self.original_ncols <= 0:
            raise ValueError("The original image dimensions must be greater than 0.")
        if self.final_nrows <=0 or self.final_ncols <= 0:
            raise ValueError("The cutted image dimensions must be greater than 0.")
        if self.nans_threshold < 0 or self.nans_threshold > 1:
            raise ValueError("The nans threshold must be between 0 and 1.")
        if self.n_images <= 0:
            raise ValueError("The number of cutted images must be greater than 0.")
        if self.final_nrows > self.original_nrows or self.final_ncols > self.original_ncols:
            raise ValueError("The cutted image dimensions must be smaller than the original image dimensions.")
        if self.total_days <= 0 or self.total_days > 28:
            raise ValueError("The total number of days must be greater than 0 and less than 28.")
        
    def _get_available_days(self, path_list: list) -> dict:
        """Get the available days in the month list

        Args:
            path_list (list): list of paths in the format path_to_folder/YYYY_MM.pt

        Returns:
            dict: dictionary with the paths path_to_folder/YYYY_MM.pt as keys and a list of the available days in the month as values
        """
        
        available_days = {}
        
        # Find the oldest and latest year, month from path.stem (YYYY_MM)
        years_months = [tuple(map(int, Path(p).stem.split("_"))) for p in path_list]
        oldest_month = min(years_months)
        latest_month = max(years_months)
        for path in path_list:
            year, month = map(int, Path(path).stem.split("_"))
            num_days = calendar.monthrange(year, month)[1]
            available_days[path] = [day for day in range(1, num_days + 1)]
            if (year, month) == oldest_month:
                available_days[path] = available_days[path][self.surrounding_days:]
            if (year, month) == latest_month:
                available_days[path] = available_days[path][:-self.surrounding_days]
            
        return available_days
    
    def get_data_paths_to_dataset_idx_dict(self, path_list: list[Path], files_dir: Path) -> tuple[dict, th.Tensor]:
        """Get a dictionary with the paths to the needed files as keys and a list (day, point, index) as values
        
        Args:
            path_list (list): list of paths to the nan masks in the format path_to_folder/YYYY_MM.pt
            files_dir (Path): path to the folder where the files are stored
            
        Returns:
            final_dict (dict): dictionary with the paths path_to_folder/YYYY_MM.pt as keys and a list of tuples (day, point, index) as values
            nan_mask_tensor (th.Tensor): tensor with the cutted masks. Shape: (n_images, final_nrows, final_ncols), 0 where nan, boolean dtype.
        """
        
        # Initialize nan masks tensor
        nan_mask_tensor = th.ones((self.n_images, self.final_nrows, self.final_ncols), dtype=th.bool)
        init_masks = th.ones((self.n_images, 13, self.final_nrows, self.final_ncols), dtype=th.bool)
        
        point = (1030, 1280)  # Starting point for the cutted images
        
        # Initialize the dictionary
        final_dict = {}
        
        mean_img_per_day = max(1, self.n_images // len(path_list))
        
        
        path_list = sorted(path_list)
        
        # Iterate over the paths
        i = 0
        k = 0
        selected_days_list = [[] for _ in range(len(path_list))]
        while i < self.n_images:
            if i < self.n_images and k == len(path_list):
                k = random.randint(0, len(path_list) - 1)  # Randomly select a path from the list if we are at the last path
            path = path_list[k]
            print(f"Processing file {path.stem}")            
            # Select the number of images for this month using a Gaussian distribution centered at mean_img_per_day
            n_days_per_file = round(random.gauss(mean_img_per_day, mean_img_per_day / 3))
            n_images_this_month = int(max(0, min(self.n_images, n_days_per_file)))
            original_nan_mask = th.load(path)
            
            min_day = 1
            max_day = original_nan_mask.shape[0]  # Assuming the first dimension is the number of days
            if path == path_list[0]:
                min_day = self.surrounding_days + 1  # Skip the first surrounding days for the first image
                print(f"day range for path {path.stem} is {min_day}-{max_day}")
            if path == path_list[-1]:
                max_day = original_nan_mask.shape[0] - self.surrounding_days
                print(f"day range for path {path.stem} is {min_day}-{max_day}")
            available_days = [day for day in range(min_day, max_day + 1)]
            
            year_month_str = Path(path).stem
            
            while n_images_this_month > 0:
                if i >= self.n_images:
                    print(f"Reached the maximum number of images: {self.n_images}. Stopping.")
                    break
                if len(available_days) <= 0:
                    print(f"No available days for path {path.stem}. Skipping this month.")
                    k += 1
                    break
                
                day = random.choice(available_days)
                
                # Get the index of the point as (x, y) coordinates and the valid cutted image

                cutted_mask = original_nan_mask[day - 1, point[0]:point[0] + self.final_nrows, point[1]:point[1] + self.final_ncols]
                mask = self.mask_class.mask()
                
                n_nans_in_mask = th.sum(~(cutted_mask | mask))
                trials = 0
                self.max_trials = 1000
                self.max_pixels = int(0.3 * self.mask_class.square_nrows * self.mask_class.square_nrows)
                while n_nans_in_mask > self.max_pixels and trials < self.max_trials:
                    mask = self.mask_class.mask()
                    n_nans_in_mask = th.sum(~(cutted_mask | mask))
                    trials += 1
                if trials == self.max_trials:
                    print("\tcould not find a valid mask")
                    continue  # Skip this day if the mask is not valid after max_trials
                
                nan_mask_tensor[i, :, :] = cutted_mask
                init_masks[i, 4, :, :] = mask
                
                year_month_day_str = f"{year_month_str}_{day:02d}"
                
                days_list = self._get_days_list(year_month_day_str)
                
                for (j, year_month_day_str) in enumerate(days_list):
                    day = int(year_month_day_str[8:])
                    file_path = files_dir / (year_month_day_str[:7] + ".pt")
                    if file_path not in list(final_dict.keys()):
                        final_dict[file_path] = []
                    
                    final_dict[file_path].append((day, point, (i, j)))
                    
                # Update the dataset index
                n_images_this_month -= 1
                selected_days_list[k].append(day)
                i += 1
            
            k += 1
        
        if i < self.n_images:
            raise ValueError(f"Could not create enough images. Only {i} images were created, but {self.n_images} were requested.")
        
        return final_dict, nan_mask_tensor, init_masks

    def _get_days_list(self, date_str: str) -> list:
        
        # Parse the input date
        date = datetime.strptime(date_str, "%Y_%m_%d").date()
        
        # Calculate previous dates
        previous_dates = [date - timedelta(days=i) for i in range(1, self.surrounding_days+1)]
        # Calculate following dates
        following_dates = [date + timedelta(days=i) for i in range(1, self.surrounding_days+1)]
        
        days_list = [d.strftime("%Y_%m_%d") for d in reversed(previous_dates)]
        days_list.append(date_str)
        days_list.extend([d.strftime("%Y_%m_%d") for d in following_dates])
                
        return days_list

    def _get_valid_nan_mask(self, original_nan_mask: th.Tensor, day: int, point: list) -> tuple:
        """Get a valid cutted nan mask from the original image, with the given point as top-left corner

        Args:
            original_nan_mask (th.Tensor): original image with the nan mask. Shape: (total_days, original_nrows, original_ncols), 0 where nan.
            day (int): day of the month to cut the image, range [1, 31]
            point (list): point as (x, y) coordinates to use as top-left corner of the cutted image

        Raises:
            ValueError: if the cutted image still has too many nans after max_trials

        Returns:
            tuple: (point, cutted_mask) where point is the top-left corner of the cutted image and cutted_mask is the cutted image. Cutted mask is 0 where nan.
        """
        
        cutted_mask = original_nan_mask[day - 1, point[0]:point[0] + self.final_nrows, point[1]:point[1] + self.final_ncols]
                
        return point, cutted_mask

    def cut(self, files_to_days_and_points_dict: dict, file_paths: list) -> th.Tensor:
        """Cut the images from the original image

        Args:
            files_to_days_and_points_dict (dict): dictionary with the paths path_to_folder/YYYY_MM.pt as keys and a list of tuples (day, point, index) as values
            file_paths (list): list of paths to the files

        Returns:
            th.Tensor: tensor with the cutted images. Shape: (n_images, n_channels, final_nrows, final_ncols)
        """
        
        n_channels = self.total_days + 4
        center_day_idx = self.surrounding_days
        
        dataset = th.ones((self.n_images, n_channels, self.final_nrows, self.final_ncols), dtype=th.float32)
        
        # print("files_to_days_and_points_dict:", files_to_days_and_points_dict)
        
        paths_list = list(files_to_days_and_points_dict.keys())
        
        lat_range = [-90, 90]
        lon_range = [-180, 180]
        scale_factor_lat = 1 / (lat_range[1] - lat_range[0])
        scale_factor_lon = 1 / (lon_range[1] - lon_range[0])

        for path in paths_list:
            print(f"Processing file {path}")
            original_file = th.load(path)
            
            points_list = files_to_days_and_points_dict[path]
            
            # print("Points list:", points_list)
            
            for (day, point, index) in points_list:
                # print(f"Processing day {day} and point {point}")
                B, C = index
                # Get the time encoding for the images
                encoded_time = self._get_encoded_time(day)
                time_layer = th.ones((self.final_nrows, self.final_ncols), dtype=th.float32) * encoded_time
                dataset[B, -1, :, :] = time_layer
                
                day_idx = day - 1
                
                # Supposing channel 0 is the SST, channel 1 is the stdev, channel 2 is the latitude and the channel 3 is the longitude
                dataset[B, C, :, :] = original_file[day_idx, 0, point[0]:point[0] + self.final_nrows, point[1]:point[1] + self.final_ncols]
                
                if C == center_day_idx:
                    # Fill the dataset with stdev, lats, lons
                    dataset[B, -4:-1, :, :] = original_file[day_idx, -3:, point[0]:point[0] + self.final_nrows, point[1]:point[1] + self.final_ncols]
                    dataset[B, -2, :, :] = 2 * (dataset[B, -2, :, :] - lat_range[0]) * scale_factor_lat - 1
                    dataset[B, -1, :, :] = 2 * (dataset[B, -1, :, :] - lon_range[0]) * scale_factor_lon - 1
                
            
            print(f"File {path} processed\n")
        return dataset     
            
    def _get_encoded_time(self, day: int) -> float:
        print(f"Encoding time for day {day}")
        norm_const = 1 / 365.25
        return math.cos(2 * math.pi * day * norm_const)

if __name__ == "__main__":
    main()