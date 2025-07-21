import math
import numpy as np
import torch as th
import xarray as xr
from pathlib import Path
import random
import math
from datetime import datetime, timedelta
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.mask_data import SquareMask

def main():
    start_time = time.time()
    N = 84 # Number of images to create
    n_days = 9
    surrounding_days = n_days // 2
    nrows = 168
    ncols = 144
    key = "sst"

    startrow = 1050
    startcol = 4600

    endrow = startrow + nrows
    endcol = startcol + ncols

    files_dir = Path("./data/modis/raw/")
    output_dir = Path("./data/datasets/dataset_1.pt")

    if not files_dir.exists():
        raise FileNotFoundError(f"Directory {files_dir} does not exist.")

    file_list = sorted(list(files_dir.glob("TERRA_MODIS.[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].L3m.DAY.NSST.sst.4km.nc")))

    if not file_list:
        raise FileNotFoundError(f"No files found in {files_dir} matching the pattern.")

    print(f"Found {len(file_list)} files.", flush=True)
    
    print("Create masks...", flush=True)
    
    dataset = {"images": None, "masks": None, "nanmasks": None, "mean": None, "std": None}
    shape = (N, n_days + 4, nrows, ncols)
    masks = th.ones(shape, dtype=th.bool)
    mc = SquareMask(image_nrows=nrows, image_ncols=ncols, mask_percentage=0.1)
    masks[:, n_days // 2, :, :] = th.stack([mc.mask() for _ in range(N)], dim=0)
    dataset["masks"] = masks
    del masks
    print("Masks created\n", flush=True)

    # Exclude the first and last n_days//2 files
    exclude = n_days // 2
    usable_files = file_list[exclude:len(file_list)-exclude]
    print(f"Using {len(usable_files)} files after excluding the first and last {exclude} files.", flush=True)
    selected_files = sorted(random.sample(usable_files, N))

    # Create a dict made as:
    # {date_str : [(dataset_idx, channel_idx), ...]}
    selected_dates = {}
    dataset_idx = 0
    for file_path in selected_files:
        date_str = file_path.stem.split(".")[1]
        if date_str not in selected_dates:
            selected_dates[date_str] = []
        
        selected_dates[date_str].append((dataset_idx, n_days // 2))  # Assuming channel index 0 for simplicity

        previous_dates, following_dates = get_surrounding_days(date_str, surrounding_days, date_format="%Y%m%d")
        
        for i, prev_date in enumerate(previous_dates):
            if prev_date not in selected_dates:
                selected_dates[prev_date] = []
            selected_dates[prev_date].append((dataset_idx, i))
        
        for i, follow_date in enumerate(following_dates):
            if follow_date not in selected_dates:
                selected_dates[follow_date] = []
            selected_dates[follow_date].append((dataset_idx, i + surrounding_days + 1))
            
        dataset_idx += 1

    # Create a tensor to hold the data
    # The number of channels is len(keys) + 4 to account for additional metadata
    # - Sin and cos of time
    # - Latitude and longitude
    images = th.ones(shape, dtype=th.float32)
    
    
    with ProcessPoolExecutor() as executor:
        futures = []
        for date_str in selected_dates:
            futures.append(executor.submit(
                process_date,
                files_dir / f"TERRA_MODIS.{date_str}.L3m.DAY.NSST.sst.4km.nc",
                selected_dates[date_str],
                startrow, endrow, startcol, endcol, n_days, key=key
            ))

        for future in as_completed(futures):
            results = future.result()
            for dataset_idx, channel_idx, arr in results:
                if channel_idx in [-4, -3] and np.isscalar(arr):
                    images[dataset_idx, channel_idx, :, :] *= arr
                else:
                    images[dataset_idx, channel_idx, :, :] = th.tensor(arr, dtype=th.float32)
        
    # for date_str in list(selected_dates.keys()):
    #     print(f"Processing date: {date_str}", flush=True)
    #     path = files_dir / f"TERRA_MODIS.{date_str}.L3m.DAY.NSST.sst.4km.nc"
    #     if not path.exists():
    #         raise FileNotFoundError(f"File {path} does not exist.")
    #     data = xr.open_dataset(path, engine="h5netcdf")

    #     for (dataset_idx, channel_idx) in selected_dates[date_str]:
    #         if channel_idx == n_days // 2:
    #             cos_time, sin_time = get_encoded_time(date_str, date_format="%Y%m%d")
    #             images[dataset_idx, -4, :, :] *= cos_time
    #             images[dataset_idx, -3, :, :] *= sin_time

    #         images[dataset_idx, channel_idx, :, :] = th.tensor(data[key].values[startrow:endrow, startcol:endcol])
    #     data.close()
        
    print("Calculating nanmasks...", flush=True)
    dataset["nanmasks"] = ~th.isnan(images)
    print("Nanmasks calculated\n", flush=True)
    
    print("Calculating mean and standard deviation...", flush=True)
    valid_nanmasks = dataset["nanmasks"][:, :n_days, :, :]
    # Find the number of non nan pixels in the sst channel
    n_valid_pixels = valid_nanmasks.float().sum().item()
    if n_valid_pixels == 0:
        raise ValueError("No valid pixels found in the dataset. Please check the input data.")

    mean = th.sum(images[:, :n_days, :, :][valid_nanmasks]).item() / n_valid_pixels
    std = th.sqrt(th.sum((images[:, :n_days, :, :][valid_nanmasks] - mean) ** 2)).item() / n_valid_pixels
    print(f"Mean: {mean}, Std: {std}\n", flush=True)
    del valid_nanmasks
    
    dataset["mean"] = mean
    dataset["std"] = std
    
    images[:, :n_days, :, :] -= mean
    images[:, :n_days, :, :] /= std
    dataset["images"] = th.nan_to_num(images, nan=-300)
    del images
        
    # Save the dataset to a file
    print()
    print(f"Saving dataset to {output_dir}", flush=True)
    th.save(dataset, output_dir)
    print(f"Dataset saved successfully in {time.time() - start_time} seconds.", flush=True)

def process_date(path, selected_tuples, startrow, endrow, startcol, endcol, n_days, key = "sst"):
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")
    date_str = path.stem.split(".")[1]
    data = xr.open_dataset(path, engine="h5netcdf")
    results = []
    for (dataset_idx, channel_idx) in selected_tuples:
        if channel_idx == n_days // 2:
            cos_time, sin_time = get_encoded_time(date_str, date_format="%Y%m%d")
            # Return time encodings for later
            results.append((dataset_idx, -4, cos_time))
            results.append((dataset_idx, -3, sin_time))
            
        arr = data[key].values[startrow:endrow, startcol:endcol]
        results.append((dataset_idx, channel_idx, arr))

    data.close()
    print(f"Processed date: {date_str}", flush=True)
    return results
    
def get_encoded_time(day_str: str, date_format = "%Y_%m_%d") -> float:
    date = datetime.strptime(day_str, date_format)
    first_day_of_year = datetime(date.year, 1, 1)
    n_days = (date - first_day_of_year).days
    norm_const = 1 / 365.25
    return math.cos(2 * math.pi * n_days * norm_const), math.sin(2 * math.pi * n_days * norm_const)

def get_surrounding_days(date_str: str, surrounding_days: int, date_format = "%Y%m%d") -> list:
   # Find previous and following n_days//2 dates for each selected date
    date_obj = datetime.strptime(date_str, date_format)
    # Calculate previous dates
    previous_dates = [date_obj - timedelta(days=i) for i in range(1, surrounding_days+1)]
    following_dates = [date_obj + timedelta(days=i) for i in range(1, surrounding_days+1)]
    
    return [date.strftime(date_format) for date in previous_dates], [date.strftime(date_format) for date in following_dates]

if __name__ == "__main__":
    main()
