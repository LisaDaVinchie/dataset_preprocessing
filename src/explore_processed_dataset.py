import torch as th
from pathlib import Path

processed_data_dir = Path("data/processed/biochemistry/")
# processed_data_dir = Path("data/processed/ensemble_physics/")
# processed_data_dir = Path("data/processed/temperature/")

processed_data_ext = ".pt"

if not processed_data_dir.exists():
    raise FileNotFoundError(f"Directory {processed_data_dir} does not exist.")

dataset_paths = sorted(processed_data_dir.glob(f"*{processed_data_ext}"))
print(f"Found {len(dataset_paths)} files in {processed_data_dir} with extension {processed_data_ext}]\n")

dataset = th.load(dataset_paths[0])

print(f"Dataset shape: {dataset.shape}\n")