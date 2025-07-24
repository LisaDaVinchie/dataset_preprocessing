import math
import numpy as np
import xarray as xr
from pathlib import Path
import random
import h5py
from datetime import datetime, timedelta
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from utils.mask_data import SquareMask, CloudMask

def main():
    start_time = time.time()
    N_train = 80
    N_test = 20
    N = N_train + N_test
    n_days = 9
    surrounding_days = n_days // 2
    nrows = 168
    ncols = 144
    key = "sst"
    batch_size = 8

    startrow = 1050
    startcol = 4600

    endrow = startrow + nrows
    endcol = startcol + ncols

    files_dir = Path("./data/modis/raw/")
    
    index = 1
    output_path = Path(f"./data/datasets/dataset_{index}.h5")
    
    while output_path.exists():
        index += 1
        output_path = Path(f"./data/datasets/dataset_{index}.h5")

    if not files_dir.exists():
        raise FileNotFoundError(f"Directory {files_dir} does not exist.")

    file_list = sorted(list(files_dir.glob("TERRA_MODIS.2010[0-9][0-9][0-9][0-9].L3m.DAY.NSST.sst.4km.nc")))

    if not file_list:
        raise FileNotFoundError(f"No files found in {files_dir} matching the pattern.")

    print(f"Found {len(file_list)} files.", flush=True)
    
    print("Create masks...", flush=True)
    
    shape = (N, n_days + 4, nrows, ncols)

    # Exclude the first and last n_days//2 files
    exclude = n_days // 2
    usable_files = file_list[exclude:len(file_list)-exclude]
    print(f"Using {len(usable_files)} files after excluding the first and last {exclude} files.", flush=True)
    selected_files = sorted(random.sample(usable_files, N))

    # Create a dict made as:
    # {date_str : [(dataset_idx, channel_idx), ...]}
    print("Selecting files...", flush=True)
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
    print(f"Selected {len(selected_dates)} dates with surrounding days.", flush=True)

    # Create a tensor to hold the data
    # The number of channels is len(keys) + 4 to account for additional metadata
    # - Sin and cos of time
    # - Latitude and longitude
    
    p = ProcessFiles(
        files_dir=files_dir,
        n_days=n_days,
        startrow=startrow,
        endrow=endrow,
        startcol=startcol,
        endcol=endcol,
        key=key)
    
    print("Dividing the work into batches...", flush=True)
    batch_dict = []
    keys = list(selected_dates.keys())
    
    for i in range(0, len(keys), batch_size):
        # print("Processing keys from", keys[i], "to", keys[min(i + batch_size, len(keys)) - 1], flush=True)
        batch = {keys[j]: selected_dates[keys[j]] for j in range(i, min(i + batch_size, len(keys)))}
        batch_dict.append(batch)

    images = np.ones(shape, dtype=np.float32)
    
    print("Add coords to images...", flush=True)
    sample_path = files_dir / f"TERRA_MODIS.{list(selected_dates.keys())[0]}.L3m.DAY.NSST.sst.4km.nc"
    data = xr.open_dataset(sample_path, engine="h5netcdf")
    lats = data["lat"].values[startrow:endrow]
    lons = data["lon"].values[startcol:endcol]
    
    print(f"Latitude shape: {lats.shape}, Longitude shape: {lons.shape}", flush=True)

    norm_lats = 2 * (lats - lats.min()) / (lats.max() - lats.min()) - 1
    norm_lons = 2 * (lons - lons.min()) / (lons.max() - lons.min()) - 1
    print(f"Normalized latitude shape: {norm_lats.shape}, Normalized longitude shape: {norm_lons.shape}", flush=True)
    
    images[:, -2, :, :] = norm_lats.repeat(ncols).reshape(nrows, ncols)
    images[:, -1, :, :] = norm_lons.repeat(nrows).reshape(ncols, nrows).T
    
    print(f"Processing {len(batch_dict)} batches of size {batch_size}...", flush=True)

    with ProcessPoolExecutor(max_workers = 4) as executor:
        futures = [
            executor.submit(p.process_batch, batch_dict[i])
            for i in range(len(batch_dict))
        ]

        # print(f"Processing {len(futures)} futures in parallel...", flush=True)
        for future in tqdm(as_completed(futures), total=len(futures)):
            results = future.result()
            # print(f"Processing {len(results)} results from the future.", flush=True)
            for dataset_idx, channel_idx, arr in results:
                # print(f"Updating {dataset_idx}, {channel_idx} with array of mean {np.nanmean(arr)} and std {np.nanstd(arr)}", flush=True)
                if channel_idx in [n_days, n_days + 1] and np.isscalar(arr):
                    images[dataset_idx, channel_idx, :, :] *= arr
                else:
                    images[dataset_idx, channel_idx, :, :] = arr
                # print(f"Updated {dataset_idx}, {channel_idx} with array of mean {np.nanmean(images[dataset_idx, channel_idx, :, :])} and std {np.nanstd(images[dataset_idx, channel_idx, :, :])}", flush=True)
       
    print("Calculating nanmasks...", flush=True)
    nanmasks = ~np.isnan(images[:, :, :, :])
    print("Nanmasks calculated\n", flush=True)
    
    print("Calculating mean and standard deviation...", flush=True)
    valid_nanmasks = nanmasks[:, :n_days, :, :]
    # Find the number of non nan pixels in the sst channel
    n_valid_pixels = valid_nanmasks.astype(float).sum().item()
    if n_valid_pixels == 0:
        raise ValueError("No valid pixels found in the dataset. Please check the input data.")

    mean = np.sum(images[:, :n_days, :, :][valid_nanmasks]) / n_valid_pixels
    std = np.sqrt(np.sum((images[:, :n_days, :, :][valid_nanmasks] - mean) ** 2) / n_valid_pixels)
    print(f"Mean: {mean}, Std: {std}\n", flush=True)
    del valid_nanmasks
    
    images[:, :n_days, :, :] -= mean
    images[:, :n_days, :, :] /= std
    images = np.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    
    masks = np.ones(shape, dtype=np.bool_)
    # mc = SquareMask(image_nrows=nrows, image_ncols=ncols, mask_percentage=0.1)
    mc = CloudMask(image_size=(nrows, ncols))
    masks[:, n_days // 2, :, :] = np.stack([mc.mask() for _ in range(N)], axis=0)
    print("Masks created\n", flush=True)
    
    print(f"Saving {len(images)} images, {len(masks)} masks, and {len(nanmasks)} nanmasks to {output_path}", flush=True)
    
    # Randomly split the indexes between 0 and N into train and test indexes
    all_indexes = list(range(N))
    random.shuffle(all_indexes)
    train_indexes = all_indexes[:N_train]
    test_indexes = all_indexes[N_train:]
    # th.save(dataset, output_dir)
    build_hdf5(output_path, 
                data_list=zip(images[train_indexes], masks[train_indexes], nanmasks[train_indexes]),
                split="train")
    
    build_hdf5(output_path, 
                data_list=zip(images[test_indexes], masks[test_indexes], nanmasks[test_indexes]), 
                split="test")

    add_metadata(output_path, mean=mean, std=std, n_days=n_days, nrows=nrows, ncols=ncols)

    # add_metadata(output_path, mean=mean, std=std, n_days=n_days)

    print(f"Dataset saved successfully in {time.time() - start_time} seconds.", flush=True)
    
def build_hdf5(h5_path, data_list, split, compression="gzip", compression_opts=4):
        with h5py.File(h5_path, 'a') as f:
            grp = f.require_group(split)
            for i, (img, mask, nanmask) in enumerate(data_list):
                sg = grp.create_group(str(i))
                sg.create_dataset('image', data=img, dtype='float32', compression=compression, compression_opts=compression_opts)
                sg.create_dataset('mask', data=mask, dtype='uint8', compression=compression, compression_opts=compression_opts)
                sg.create_dataset('nanmask', data=nanmask, dtype='uint8', compression=compression, compression_opts=compression_opts)

def add_metadata(h5_path, mean, std, n_days, nrows, ncols):
    with h5py.File(h5_path, 'a') as f:
        f.attrs['mean'] = mean
        f.attrs['std'] = std
        f.attrs['n_days'] = n_days
        f.attrs['nrows'] = nrows
        f.attrs['ncols'] = ncols


class ProcessFiles:
    def __init__(self, files_dir: Path, n_days: int, startrow: int, endrow: int, startcol: int, endcol: int, key: str = "sst"):
        self.files_dir = files_dir
        self.n_days = n_days
        self.startrow = startrow
        self.endrow = endrow
        self.startcol = startcol
        self.endcol = endcol
        self.key = key

    def process_batch(self, dates_dict: dict):
        results = []
        for date_str in list(dates_dict.keys()):
            
            path = self.files_dir / f"TERRA_MODIS.{date_str}.L3m.DAY.NSST.sst.4km.nc"
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist.")
            data = xr.open_dataset(path, engine="h5netcdf")
            for (dataset_idx, channel_idx) in dates_dict[date_str]:
                # print(f"Processing date: {date_str}, dataset_idx: {dataset_idx}, channel_idx: {channel_idx}", flush=True)
                if channel_idx == self.n_days // 2:
                    cos_time, sin_time = get_encoded_time(date_str, date_format="%Y%m%d")
                    # Return time encodings for later
                    results.append((dataset_idx, self.n_days, cos_time))
                    results.append((dataset_idx, self.n_days + 1, sin_time))
                    
                arr = data[self.key].values[self.startrow:self.endrow, self.startcol:self.endcol]

                results.append((dataset_idx, channel_idx, arr))

            data.close()
            # print(f"Processed date: {date_str}\n", flush=True)
        
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
