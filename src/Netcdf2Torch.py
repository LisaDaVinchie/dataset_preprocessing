import xarray as xr
import torch as th
import time
from pathlib import Path
import argparse
import json

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Convert NetCDF to Torch')
    parser.add_argument('--paths', type=Path, help='File containing paths', required=True)
    parser.add_argument('--params', type=Path, help='Output directory', required=True)

    args = parser.parse_args()
    paths_file = Path(args.paths)

    with open(paths_file, "r") as f:
        paths = json.load(f)

    raw_data_dir = Path(paths["data"]["raw_data_dir"])
    processed_data_dir = Path(paths["data"]["processed_data_dir"])
    processed_data_ext = paths["data"]["processed_data_ext"]

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Directory {raw_data_dir} does not exist.")
    
    # Create the processed data directory if it doesn't exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    raw_data_paths = list(raw_data_dir.glob("*.nc"))

    params_path = Path(args.params)

    # Load the parameters
    with open(params_path, "r") as f:
        params = json.load(f)

    channels_to_keep = list(params["dataset"]["channels_to_keep"])
    n_rows = int(params["dataset"]["n_rows"])
    n_cols = int(params["dataset"]["n_cols"])
    dataset_name = str(params["dataset"]["dataset_name"])
    
    year_position = list(params[dataset_name]["year_position"])
    month_position = list(params[dataset_name]["month_position"])
    day_position = list(params[dataset_name]["day_position"])
    
    if year_position is None or month_position is None or day_position is None:
        raise ValueError("Year, month or day position is None")

    print(f"Keys to include: {channels_to_keep}\n")

    # Assuming all the files have the same keys and shape

    print(f"Processing {len(raw_data_paths)} files")
    for file in raw_data_paths:
        print(f"Processing {file.name}")
        
        processed_data_path = generate_processed_data_path(processed_data_dir, processed_data_ext, file, year_position, month_position, day_position)
        
        output_tensor = file_to_tensor(processed_data_dir, processed_data_ext, channels_to_keep, n_rows, n_cols, file)
        if output_tensor is None:
            continue

        th.save(output_tensor, processed_data_path)
        
    elapsed_time = time.time() - start_time

    print(f"Elapsed time: {elapsed_time}")

def generate_processed_data_path(processed_data_dir: Path, processed_data_ext: str, file_path: Path, year_position: list, month_position: list, day_position: list) -> Path:
    """Gernerate the processed data path from the raw data path.

    Args:
        processed_data_dir (Path): path to the processed data directory
        processed_data_ext (str): extension of the processed data files
        file_path (Path): path to the raw data file
        year_position (list): position of the year in the file name, as a list of integers [start, end]
        month_position (list): position of the month in the file name, as a list of integers [start, end]
        day_position (list): position of the day in the file name, as a list of integers [start, end]

    Returns:
        Path: path to the processed data file
    """
    file_name = file_path.name
    year = file_name[year_position[0]:year_position[1]]
    month = file_name[month_position[0]:month_position[1]]
    day = file_name[day_position[0]:day_position[1]]
    processed_data_path = processed_data_dir / f"{year}_{month}_{day}{processed_data_ext}"
    return processed_data_path

def file_to_tensor(keys_to_keep: list, n_rows: int, n_cols: int, file_path: Path) -> tuple[Path, th.Tensor]:
    """Convert a single file to a tensor.

    Args:
        keys_to_keep (list): list of the names of the dataset keys to keep
        n_rows (int): number of rows in the image
        n_cols (int): number of columns in the image
        file_path (Path): path to the file to convert

    Returns:
        th.Tensor: tensor of the data
    """
    output_tensor = th.zeros(len(keys_to_keep), n_rows, n_cols)
    try:
        data = xr.open_dataset(file_path, engine="h5netcdf")
            
        for i, key in enumerate(keys_to_keep):
            output_tensor[i, :, :] = th.tensor(data[key].values).unsqueeze(1)
                
        data.close()
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}", flush=True)
        return None
    
    return output_tensor
    

if __name__ == "__main__":
    main()