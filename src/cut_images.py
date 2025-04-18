import torch as th
from pathlib import Path
import argparse
from time import time
from datetime import datetime
import random
import json
import os
import sys
import math

path_to_append = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_to_append)
from mask_data import mask_inversemask_image, create_square_mask

class CutAndMaskImage:
    """Class used to:
        - select random smaller images from a larger image
        - mask some pixels in the smaller images to simulate missing data
        - adding a time layer to the images, indicating the date
        - save the dataset and masks
        - save info regarding the dataset, such as the number of images, the size of the images, etc.
    """
    def __init__(self, original_nrows: int, original_ncols: int, final_nrows: int, final_ncols: int, nans_threshold: float, n_cutted_images: int):
        """Initalize the class with the parameters needed to cut the images and generate the dataset

        Args:
            original_nrows (int): number of rows in the original image
            original_ncols (int): number of columns in the original image
            final_nrows (int): number of rows in the cutted image
            final_ncols (int): number of columns in the cutted image
            nans_threshold (float): maximum fraction of nans allowed in the cutted image. Must be between 0 and 1
            n_cutted_images (int): number of cutted images to generate
        """
        self.original_nrows = original_nrows
        self.original_ncols = original_ncols
        self.final_nrows = final_nrows
        self.final_ncols = final_ncols
        self.nans_threshold = nans_threshold
        self.n_cutted_images = n_cutted_images


    def select_random_points(self, n_points: int) -> th.Tensor:
        """Select random points in the original image, to use as top-left corners for the cutted images

        Args:
            n_points (int): number of random points to select

        Returns:
            th.Tensor: tensor with the selected random points, as (x, y) coordinates
        """
        random_x = th.randint(0, self.original_nrows - self.final_nrows, (n_points,))
        random_y = th.randint(0, self.original_ncols - self.final_ncols, (n_points,))
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
        cutted_img = image[:, index[0]:index[0] + self.final_nrows, index[1]:index[1] + self.final_ncols]
        nan_count = th.isnan(cutted_img).sum().item()
        if nan_count > n_pixel_threshold:
            while nan_count > n_pixel_threshold:
                index = self.select_random_points(1)[0]
                cutted_img = image[:, index[0]:index[0] + self.final_nrows, index[1]:index[1] + self.final_ncols]
                nan_count = th.isnan(cutted_img).sum().item()
        return cutted_img

    def generate_image_dataset(self, n_channels: int, masked_fraction: float, masked_channels_list: list, path_to_indices_map: dict, extended_data: bool, minimal_data: bool, placeholder: float = None) -> tuple[dict, dict, th.Tensor]:
        """Generate a dataset of masked images, inverse masked images and masks

        Args:
            n_channels (int): final number of channels in the image
            masked_fraction (float): percentage of masked pixels in each channel
            masked_channels_list (list): list of channels that should not be masked
            path_to_indices_map (dict): dictionary with the paths to the images as keys and the points as values
            extended_data (bool): True if the extended dataset should be generated
            minimal_data (bool): True if the minimal dataset should be generated
            placeholder (float): value to use as placeholder for masked pixels, None if the mean of the image should be used

        Returns:
            tuple: dictionary with the masked images, inverse masked images and masks as keys and the corresponding tensors as values
                    and dictionary with the images and masks as keys and the corresponding tensors as values
        """
        
        # Initialize the two datasets
        dataset_ext, dataset_min = None, None
        if extended_data:
            keys_ext = ["masked_images", "inverse_masked_images", "masks"]
            dataset_ext = {cls: th.empty((self.n_cutted_images, n_channels, self.final_nrows, self.final_ncols), dtype=th.float32) for cls in keys_ext}
        if minimal_data:
            keys_min = ["images", "masks"]
            dataset_min = {cls: th.empty((self.n_cutted_images, n_channels, self.final_nrows, self.final_ncols), dtype=th.float32) for cls in keys_min}
            
        nans_masks = th.ones((self.n_cutted_images, n_channels, self.final_nrows, self.final_ncols), dtype=th.float32)
        
        masked_channels_list = list(masked_channels_list)
        
        # Calculate the interval between the oldest and newest date, in days
        selected_dates = [Path(path).stem for path in path_to_indices_map.keys()]
        dates_parsed = [datetime.strptime(d, "%Y_%m_%d").date() for d in selected_dates]
        oldest_date, newest_date = min(dates_parsed), max(dates_parsed)
        interval = (newest_date - oldest_date).days

        idx_start = 0 # index of the first cutted image of this raw image
        idx_end = 0 # index of the last cutted image of this raw image
        n_original_channels = n_channels - 1 # The last channel is the time layer
        n_pixels = self.final_nrows * self.final_ncols * n_original_channels # number of pixels in the raw image
        threshold = self.nans_threshold * n_pixels # threshold of nans in the image
        for path, indices in path_to_indices_map.items():
            image = th.load(path)
            n_indices = len(indices)
            idx_end = idx_start + n_indices
            
            # Add the time layer to the images
            days_from_inital_date = (datetime.strptime(Path(path).stem, "%Y_%m_%d").date() - oldest_date).days
            encoded_time = math.cos(math.pi * days_from_inital_date / interval)
            
            # Generate the cutted images, adding the time layer
            cutted_imgs = th.stack([self.cut_valid_image(image, index, threshold) for index in indices], dim=0)
            
            time_layers = th.ones((n_indices, 1, self.final_nrows, self.final_ncols), dtype=th.float32) * encoded_time
            cutted_imgs = th.cat((cutted_imgs, time_layers), dim=1)
            
            # Find where the nans are in the cutted images
            cutted_img_nans = ~th.isnan(cutted_imgs)
            nans_masks[idx_start:idx_end, :, :, :] = cutted_img_nans.float()
            
            # Create square masks. 0 where the values are masked, 1 where the values are not masked
            masks = th.ones((n_indices, n_channels, self.final_nrows, self.final_ncols), dtype=th.float32)
            for mc in masked_channels_list:
                masks[:, mc, :, :] = create_square_mask(self.final_nrows, self.final_ncols, masked_fraction)
            
            # Set masks to 0 where the nan mask is 0
            masks = th.where(cutted_img_nans == 0, th.tensor(0, dtype=masks.dtype), masks)
            
            # Save the images to the minimal dataset, substituting the nans with the placeholder
            if minimal_data:
                dataset_min[keys_min[0]][idx_start:idx_end] = th.nan_to_num(cutted_imgs, nan=placeholder)
                dataset_min[keys_min[1]][idx_start:idx_end] = masks
            
            # Save the images to the extended dataset
            if extended_data:
                masked_images, inverse_masked_images = mask_inversemask_image(cutted_imgs, masks, placeholder)
                dataset_ext[keys_ext[0]][idx_start:idx_end] = masked_images
                dataset_ext[keys_ext[1]][idx_start:idx_end] = inverse_masked_images
                dataset_ext[keys_ext[2]][idx_start:idx_end] = masks
            
            idx_start = idx_end
            
        return dataset_ext, dataset_min, nans_masks

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

def main():
    start_time = time()
    parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
    parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
    parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')

    args = parser.parse_args()

    params_path = args.params
    paths_path = args.paths
    
    with open(paths_path, 'r') as json_file:
        json_paths = json.load(json_file)
        
    processed_data_dir = Path(json_paths["data"]["processed_data_dir"])
    next_extended_dataset_path = Path(json_paths["data"]["next_extended_dataset_path"])
    next_minimal_dataset_path = Path(json_paths["data"]["next_minimal_dataset_path"])
    dataset_specs_path = Path(json_paths["data"]["dataset_specs_path"])
    next_nans_masks_path = Path(json_paths["data"]["next_nans_masks_path"])
    
    # Check if the directories exist
    check_dirs_existance([processed_data_dir, next_extended_dataset_path.parent, next_minimal_dataset_path.parent, dataset_specs_path.parent])

    with open(params_path, 'r') as json_file:
        params = json.load(json_file)
    
    x_shape_raw = int(params["dataset"]["x_shape_raw"])
    y_shape_raw = int(params["dataset"]["y_shape_raw"])
    n_cutted_images = int(params["dataset"]["n_cutted_images"])
    cutted_nrows = int(params["dataset"]["cutted_nrows"])
    cutted_ncols = int(params["dataset"]["cutted_ncols"])
    n_channels = int(params["dataset"]["n_channels"])
    masked_channels = list(params["dataset"]["masked_channels"])
    nans_threshold = float(params["dataset"]["nans_threshold"])
    minimal_dataset = bool(params["dataset"]["minimal_dataset"])
    extended_dataset = bool(params["dataset"]["extended_dataset"])
    
    mask_percentage = float(params["mask"]["mask_percentage"])
    placeholder = float(params["mask"]["placeholder"])
    
    if minimal_dataset == False and extended_dataset == False:
        raise ValueError("Both minimal_dataset and extended_dataset are False. At least one of them should be True.")

    if placeholder == False:
        placeholder = None
        
    print("Using placeholder value: ", placeholder, flush=True)

    # Select n_images random images from the processed images
    processed_images_paths = list(processed_data_dir.glob(f"*.pt"))

    if len(processed_images_paths) == 0:
        raise FileNotFoundError(f"No images found in {processed_data_dir}")

    print(f"\nFound {len(processed_images_paths)} images in {processed_data_dir}\n", flush=True)
    
    cut_class = CutAndMaskImage(original_nrows=x_shape_raw, original_ncols=y_shape_raw,
                                final_nrows=cutted_nrows, final_ncols=cutted_ncols,
                                nans_threshold=nans_threshold, n_cutted_images=n_cutted_images)

    # Select some random points, to use as centers for the cutted images
    idx_time = time()
    random_points = cut_class.select_random_points(n_points=n_cutted_images)
    print(f"Selected random points for cutted images in {time() - idx_time} seconds\n", flush=True)

    path_to_indices = cut_class.map_random_points_to_images(processed_images_paths, random_points)
    
    print(f"Mapped the points to images in {time() - idx_time} seconds\n", flush=True)

    d_time = time() 
    # Generate the dataset
    dataset_ext, dataset_min, nans_masks = cut_class.generate_image_dataset(n_channels=n_channels,
                                                    masked_fraction=mask_percentage, masked_channels_list=masked_channels,
                                                    path_to_indices_map=path_to_indices, minimal_data=minimal_dataset,
                                                    extended_data=extended_dataset, placeholder=placeholder)
    print(f"Generated the dataset in {time() - d_time} seconds\n", flush=True)
    
    if dataset_ext is not None:
        th.save(dataset_ext, next_extended_dataset_path)
        print(f"Saved the extended dataset to {next_extended_dataset_path}\n", flush=True)
    if dataset_min is not None:
        th.save(dataset_min, next_minimal_dataset_path)
        print(f"Saved the minimal dataset to {next_minimal_dataset_path}\n", flush=True)
        
    th.save(nans_masks, next_nans_masks_path)
    print(f"Saved the nans masks to {next_nans_masks_path}\n", flush=True)

    # Extract the "dataset" and "mask" sections
    dataset_section = json_paths.get('dataset', {})
    mask_section = json_paths.get('mask', {})

    # Combine the sections into a single dictionary
    sections_to_save = {
        'dataset': dataset_section,
        'mask': mask_section
    }

    elapsed_time = time() - start_time
    
    # Save the combined sections to a text file
    with open(dataset_specs_path, 'w') as txt_file:
        txt_file.write("# Dataset specifications\n")
        txt_file.write(f"Elapsed time:\n{elapsed_time:.2f} seconds\n")
        txt_file.write("Dataset and mask specifications:\n")
        json.dump(sections_to_save, txt_file, indent=4)

    print("Elapsed time: {:.2f} seconds\n".format(time() - start_time), flush=True)
    
if __name__ == "__main__":
    main()