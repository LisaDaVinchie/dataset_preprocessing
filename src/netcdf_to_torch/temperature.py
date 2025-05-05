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
    
    netcdftotorch = NetcdfToTorch(raw_data_dir, processed_data_dir, processed_data_ext, params_path)

    print(f"Keys to include: {netcdftotorch.keys_to_keep}\n")
    print(f"Number of keys: {len(netcdftotorch.keys_to_keep)}\n")

    # Assuming all the files have the same keys and shape

    print(f"Processing {len(raw_data_paths)} files")
    for file in raw_data_paths:
        print(f"Processing {file.name}")
        
        processed_data_path = netcdftotorch.generate_processed_data_path(file)
        
        output_tensor = netcdftotorch.file_to_tensor(file)
        if output_tensor is None:
            continue

        th.save(output_tensor, processed_data_path)
        print("Tensor saved to", processed_data_path)
        
    elapsed_time = time.time() - start_time

    print(f"Elapsed time: {elapsed_time}")
 
class NetcdfToTorch:
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path, processed_data_ext: str, params_path: Path):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.processed_data_ext = processed_data_ext
        self.params_path = params_path
        self.n_days = 31
        
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
        
        n_keys = len(self.keys_to_keep)
        output_tensor = th.zeros(self.n_days, n_keys, self.n_rows, self.n_cols)
        print(f"Output tensor shape: {output_tensor.shape}")
        try:
            data = xr.open_dataset(file_path, engine="h5netcdf")
            print("Dataset opened")
            for i, key in enumerate(self.keys_to_keep):
                images = th.tensor(data[key].values)
                output_tensor[:, i, :, :] = images
                    
            data.close()
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}", flush=True)
            return None
        
        return output_tensor
    

if __name__ == "__main__":
    main()