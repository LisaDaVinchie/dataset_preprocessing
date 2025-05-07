import xarray as xr
import torch as th
import time
from pathlib import Path
import argparse
import json
import re
from collections import defaultdict
import calendar
import re

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Convert NetCDF to Torch')
    parser.add_argument('--paths', type=Path, help='File containing paths', required=True)
    parser.add_argument('--params', type=Path, help='Output directory', required=True)

    args = parser.parse_args()
    paths_file = Path(args.paths)

    with open(paths_file, "r") as f:
        paths = json.load(f)
        
    dir_name = "biochemistry"

    raw_data_dir = Path(paths[dir_name]["raw_data_dir"])
    processed_data_dir = Path(paths[dir_name]["processed_data_dir"])
    processed_data_ext = paths["processed_data_ext"]

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Directory {raw_data_dir} does not exist.")
    
    # Create the processed data directory if it doesn't exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    sub_dirs = [d for d in raw_data_dir.iterdir() if d.is_dir()]

    raw_data_paths = []
    
    pattern = re.compile(r"^\d{4}_\d{2}\.nc$")
    
    for dir in sub_dirs:
        raw_data_paths.append(sorted([path for path in dir.glob("*.nc") if pattern.match(path.name)]))
        
    print(f"Found {len(raw_data_paths)} directories with {len(raw_data_paths[0])} files each")
    
    paths_by_date = group_files_by_date(raw_data_paths)
    print(f"Found {len(paths_by_date)} unique dates")

    params_path = Path(args.params)
    
    # Load the parameters
    with open(params_path, "r") as f:
        params = json.load(f)
        
    
    dataset_name = str(params["dataset"]["dataset_name"])
    keys_to_keep = list(params[dataset_name]["channels_to_keep"])
    n_rows = int(params[dataset_name]["n_rows"])
    n_cols = int(params[dataset_name]["n_cols"])
    
    print(f"Dates: {list(paths_by_date.keys())}\n")
    
    for date in list(paths_by_date.keys()):
        print(f"Processing date {str(date)}", flush=True)
        file_name = Path(date).stem + processed_data_ext
        processed_data_path = processed_data_dir / file_name
        
        n_days = calculate_days_in_month(date)
        
        try:
            output_tensor = th.zeros(n_days, len(keys_to_keep), n_rows, n_cols)
            idx = 0
            for path in paths_by_date[date]: # For each file
                data = xr.open_dataset(path, engine="h5netcdf")
                
                data_keys = list(data.keys())
                
                # Find data keys that are in keys_to_keep
                valid_keys = [key for key in data_keys if key in keys_to_keep]
                if not valid_keys:
                    print(f"No matching keys found in {path}. Skipping file.")
                    continue
                    
                for key in valid_keys:
                    images = data[key]
                    
                    if images.ndim == 4: # Multy channel
                        images = images[:, 0, :, :].values
                    elif images.ndim == 3: # Single channel
                        images = images.values
                    
                    output_tensor[:, idx, :, :] = th.tensor(images)
                    idx += 1
                        
                data.close()
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
        
        # Save the tensor to a file
        th.save(output_tensor, processed_data_path)

        print()
        
    elapsed_time = time.time() - start_time

    print(f"Elapsed time: {elapsed_time}")
    
def calculate_days_in_month(file_name: str) -> int:
        """Get the number of days in the month from the file name.

        Args:
            file_name (str): file name in the format YYYY_MM.nc

        Returns:
            int: number of days in the month
        """
        month = int(file_name.split("_")[1])
        year = int(file_name.split("_")[0])
        
        days_in_month = calendar.monthrange(year, month)[1]
        return days_in_month
    
def group_files_by_date(files):
    """Group files by date in the format YYYYMM.

    Args:
        files (list): List of file paths.

    Returns:
        dict: Dictionary with date as key and list of file paths as value.
    """

    # Step 1: Flatten the list (if needed) and extract dates
    date_to_paths = defaultdict(list)

    for path_sublist in files:
        for path in path_sublist:
            # Extract date from the filename (stem removes '.nc')
            match = re.search(r"(\d{4}_\d{2})$", path.stem)
            if match:
                date = match.group(1)
                date_to_paths[date].append(path)
    return dict(date_to_paths)

if __name__ == "__main__":
    main()