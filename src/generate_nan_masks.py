import torch as th
from pathlib import Path
import time

from utils.parse_params import parse_input_paths

start_time = time.time()

params, paths = parse_input_paths()

# Paths to the input data
dataset_kind = str(params["dataset"]["dataset_kind"])
processed_data_dir = Path(paths[dataset_kind]["processed_data_dir"])
original_nan_masks_dir = Path(paths[dataset_kind]["nan_masks_dir"])

if not processed_data_dir.exists():
    raise ValueError(f"Processed data directory {processed_data_dir} does not exist.")
if not original_nan_masks_dir.exists():
    raise ValueError(f"Original nan masks directory {original_nan_masks_dir} does not exist.")

original_data_paths = list(processed_data_dir.glob("*.pt"))

for path in original_data_paths:
    print(f"Processing {path.name}...")
    # Load the original data
    original_data = th.load(path)
    # Get the corresponding nan mask path
    nan_mask_path = original_nan_masks_dir / path.name
    
    nan_mask = ~th.isnan(original_data[:, 0, :, :])
    
    # Save the nan mask
    th.save(nan_mask, nan_mask_path)
    
    print(f"Saved nan mask to {nan_mask_path}\n")
    
print("All nan masks generated and saved.")
print("Elapsed time: ", time.time() - start_time)