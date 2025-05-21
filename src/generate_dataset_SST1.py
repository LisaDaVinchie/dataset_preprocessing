import torch as th
from pathlib import Path
import pickle
import math
import random
import calendar
from datetime import datetime, timedelta
from time import time
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
    
    # Get the paths to the files
    original_nan_masks_paths = list(original_nan_masks_dir.glob("*.pt"))
    if len(original_nan_masks_paths) == 0:
        raise ValueError(f"No files found in {original_nan_masks_dir}.")
    original_nan_masks_paths.sort()
    
    # Map the random points to the days
    
    final_dict, nan_mask_tensor = cut.get_data_paths_to_dataset_idx_dict(original_nan_masks_paths, processed_data_dir)
    
    # Get the paths to the files
    original_data_paths = list(processed_data_dir.glob("*.pt"))
    if len(original_data_paths) == 0:
        raise ValueError(f"No files found in {processed_data_dir}.")
    original_data_paths.sort()
    
    # Cut the images
    cutted_images = cut.cut(final_dict, original_data_paths)
    
    nan_masks = ~th.isnan(cutted_images)
    
    dataset = {}
    dataset["images"] = th.nan_to_num(cutted_images, nan=nan_placeholder)
    
    dataset["masks"] = create_masks(params, cutted_images, nan_masks)
    
    # Save the cutted images
    th.save(dataset, dataset_path)
    
    # Save the nan masks
    th.save(nan_masks, nanmasks_path)
    
    print("Elapsed time for parsing input paths: ", time() - start_time)

def create_masks(params, cutted_images, nan_masks):
    masks = th.ones_like(cutted_images, dtype=th.bool)
    
    dataset_kind = str(params["dataset"]["dataset_kind"])
    masked_channels = list(params["dataset"][dataset_kind]["masked_channels"])
    
    mask_class = initialize_mask_kind(params)
    n_images = cutted_images.shape[0]
    for j in masked_channels:
        masks[:, j, :, :] = th.stack([mask_class.mask() for _ in range(n_images)], dim=0)
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
        
    def _load_parameters(self, params):
        if params is not None:
            dataset_params = params["dataset"]
            dataset_kind = str(dataset_params["dataset_kind"])
            if self.original_nrows is None:
                self.original_nrows = int(dataset_params[dataset_kind]["n_rows"])
            if self.original_ncols is None:
                self.original_ncols = int(dataset_params[dataset_kind]["n_cols"])
            if self.final_nrows is None:
                self.final_nrows = int(dataset_params[dataset_kind]["final_n_rows"])
            if self.final_ncols is None:
                self.final_ncols = int(dataset_params[dataset_kind]["final_n_cols"])
            if self.n_images is None:
                self.n_images = int(dataset_params[dataset_kind]["n_images"])
            if self.nans_threshold is None:
                self.nans_threshold = float(dataset_params[dataset_kind]["nans_perc"])
            if self.total_days is None:
                self.total_days = int(dataset_params[dataset_kind]["total_days"])
        
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

        for path in path_list:
            year, month = map(int, Path(path).stem.split("_"))
            num_days = calendar.monthrange(year, month)[1]
            available_days[path] = [day for day in range(1, num_days + 1)]
        return available_days
    
    def _select_random_points(self, n_points: int) -> list[tuple]:
        """Select random points in the original image, to use as top-left corners for the cutted images

        Args:
            n_points (int): number of random points to select

        Returns:
            list: list of tuples with the random points as (x, y) coordinates
        """
        
        max_x = self.original_nrows - self.final_nrows
        max_y = self.original_ncols - self.final_ncols
        
        random_points = [(x, y) for x, y in zip(random.sample(range(0, max_x), n_points), random.sample(range(0, max_y), n_points))]
        return random_points
    
    def _map_random_points_to_days(self, path_list: list) -> dict:
        """Assign the points to some random images

        Args:
            path_list (list): list of paths to the nan masks in the format path_to_folder/YYYY_MM.pt

        Returns:
            dict: dictionary with the paths path_to_folder/YYYY_MM.pt as keys and a list of tuples (day, point) as values
        """
        
        days_list = self._get_available_days(path_list)
        paths = list(days_list.keys())
        
        points_list = self._select_random_points(self.n_images)
        
        points_to_days = {}
        
        for point in points_list:
            path = random.choice(paths)
            day = random.choice(days_list[path])
            if path not in points_to_days:
                points_to_days[path] = []
            points_to_days[path].append((day, point))
        
        return points_to_days
    
    def get_data_paths_to_dataset_idx_dict(self, path_list: list[Path], files_dir: Path) -> tuple[dict, th.Tensor]:
        """Get a dictionary with the paths to the needed files as keys and a list (day, point, index) as values
        
        Args:
            path_list (list): list of paths to the nan masks in the format path_to_folder/YYYY_MM.pt
            files_dir (Path): path to the folder where the files are stored
            
        Returns:
            final_dict (dict): dictionary with the paths path_to_folder/YYYY_MM.pt as keys and a list of tuples (day, point, index) as values
            nan_mask_tensor (th.Tensor): tensor with the cutted masks. Shape: (n_images, final_nrows, final_ncols), 0 where nan, boolean dtype.
        """
        
        points_to_days_dict = self._map_random_points_to_days(path_list)
        
        paths_list = list(points_to_days_dict.keys())
        # Initialize nan masks tensor
        nan_mask_tensor = th.ones((self.n_images, self.final_nrows, self.final_ncols), dtype=th.bool)
        
        # Initialize the dictionary
        final_dict = {}
        
        # Index of the image in the future dataset
        i = 0
        
        # Iterate over the paths
        for path in paths_list:
            original_nan_mask = th.load(path)
            
            day_to_point_list = points_to_days_dict[path]
            
            month_str = Path(path).stem
            
            data_file_path = files_dir / f"{month_str}.pt"
            
            final_dict[data_file_path] = {}
            
            dict_list = []
            
            for day, point in day_to_point_list:
                # Get the index of the point as (x, y) coordinates and the valid cutted image
                point, cutted_mask = self._get_valid_nan_mask(original_nan_mask, day, point)
                
                # Add the cutted image to the tensor
                nan_mask_tensor[i, :, :] = cutted_mask
                
                day_str = f"{month_str}_{day:02d}"
                
                days_list = self._get_days_list(day_str)
                
                for (k, day_str) in enumerate(days_list):
                    day = int(day_str[8:])
                    dict_list.append((day, point, (i, k)))
                    
                # Update the dataset index
                i += 1
                
            final_dict[data_file_path] = dict_list
        
        return final_dict, nan_mask_tensor 

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
                
        n_nans = th.sum(~cutted_mask).float()
                
        # Check if the cutted image has too many nans
        if n_nans > self.max_pixels:
            trials = 0
            while n_nans > self.max_pixels and trials < self.max_trials:
                # Get a new point
                point = self._select_random_points(1)[0]
                cutted_mask = original_nan_mask[day - 1, point[0]:point[0] + self.final_nrows, point[1]:point[1] + self.final_ncols]
                n_nans = th.sum(~cutted_mask).float()
                trials += 1
            if trials == self.max_trials:
                raise ValueError(f"Could not find a valid cutted image after {trials} trials. The image has too many nans.")
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
        
        paths_list = list(files_to_days_and_points_dict.keys())
        

        for path in paths_list:
            original_file = th.load(path)
            
            points_list = files_to_days_and_points_dict[path]
            
            for (day, point, index) in points_list:
                
                B, C = index
                # Get the time encoding for the images
                encoded_time = self._get_encoded_time(day)
                time_layer = th.ones((self.final_nrows, self.final_ncols), dtype=th.float32) * encoded_time
                dataset[B, -1, :, :] = time_layer
                
                day -= 1
                
                # Supposing channel 0 is the SST, channel 1 is the stdev, channel 2 is the latitude and the channel 3 is the longitude
                dataset[B, C, :, :] = original_file[day, 0, point[0]:point[0] + self.final_nrows, point[1]:point[1] + self.final_ncols]
                
                if C == center_day_idx:
                    # Fill the dataset with stdev, lats, lons
                    dataset[B, -4:-1, :, :] = original_file[day, -3:, point[0]:point[0] + self.final_nrows, point[1]:point[1] + self.final_ncols]
                    
        return dataset     
            
    def _get_encoded_time(self, day: int) -> float:
        norm_const = 1 / 365.25
        return math.cos(2 * math.pi * day * norm_const)

if __name__ == "__main__":
    main()