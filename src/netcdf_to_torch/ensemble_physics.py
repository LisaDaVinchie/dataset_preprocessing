import xarray as xr
import torch as th
import time
from pathlib import Path
import argparse
import json
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
        
    dir_name = "ensemble_physics"

    raw_data_dir = Path(paths[dir_name]["raw_data_dir"])
    processed_data_dir = Path(paths[dir_name]["processed_data_dir"])
    processed_data_ext = paths["processed_data_ext"]

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Directory {raw_data_dir} does not exist.")
    
    # Create the processed data directory if it doesn't exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"^\d{4}_\d{2}\.nc$")
    raw_data_paths = [file for file in list(raw_data_dir.glob("*.nc")) if pattern.match(file.name)]

    params_path = Path(args.params)
    
    netcdftotorch = NetcdfToTorch(raw_data_dir, processed_data_dir, processed_data_ext, params_path)

    print(f"Keys to include: {netcdftotorch.keys_to_keep}\n")

    # Assuming all the files have the same keys and shape

    print(f"Processing {len(raw_data_paths)} files")
    for file in raw_data_paths:
        print(f"Processing {file.name}")
        
        # Check if the file name matches the expected format YYYY_MM.nc
        if not file.name.endswith(".nc") or len(file.stem.split("_")) != 2 or not file.stem.split("_")[0].isdigit() or not file.stem.split("_")[1].isdigit():
            print(f"Skipping {file.name}: Invalid file format. Expected format is YYYY_MM.nc")
            continue
        
        processed_data_path = netcdftotorch.generate_processed_data_path(file)
        
        output_tensor = netcdftotorch.file_to_tensor(file)
        if output_tensor is None:
            continue

        th.save(output_tensor, processed_data_path)
        
    elapsed_time = time.time() - start_time

    print(f"Elapsed time: {elapsed_time}")
class NetcdfToTorch:
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path, processed_data_ext: str, params_path: Path):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.processed_data_ext = processed_data_ext
        self.params_path = params_path
        self.n_days = self.calculate_days_in_month(self.raw_data_dir.stem)
        
        # Load the parameters
        self.load_params()
    
    def load_params(self):
        # Load the parameters
        with open(self.params_path, "r") as f:
            params = json.load(f)
            
        
        dataset_name = str(params["dataset"]["dataset_name"])
        self.keys_to_keep = list(params[dataset_name]["channels_to_keep"])
        self.n_rows = int(params[dataset_name]["n_rows"])
        self.n_cols = int(params[dataset_name]["n_cols"])
    
    def calculate_days_in_month(self, file_name: str) -> int:
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
        
    def generate_processed_data_path(self, file_path: Path) -> Path:
        """Gernerate the processed data path from the raw data path.

        Args:
            file_path (Path): path to the raw data file

        Returns:
            Path: path to the processed data file
        """
        file_name = file_path.stem
        processed_data_path = self.processed_data_dir / f"{file_name}{self.processed_data_ext}"
        return processed_data_path

    def file_to_tensor(self, file_path: Path) -> tuple[Path, th.Tensor]:
        """Convert a single file to a tensor.

        Args:
            file_path (Path): path to the file to convert

        Returns:
            th.Tensor: tensor of the data
        """
        output_tensor = th.zeros(self.n_days, len(self.keys_to_keep), self.n_rows, self.n_cols)
        try:
            data = xr.open_dataset(file_path, engine="h5netcdf")
                
            for i, key in enumerate(self.keys_to_keep):
                images = data[key]
                
                if images.ndim == 4: # Multy channel
                    images = images[:, 0, :, :].values
                elif images.ndim == 3: # Single channel
                    images = images[:, :, :].values
                
                output_tensor[:, i, :, :] = th.tensor(images)
                    
            data.close()
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}", flush=True)
            return None
        
        return output_tensor
    

if __name__ == "__main__":
    main()