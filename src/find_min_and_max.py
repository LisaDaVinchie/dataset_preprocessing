import torch as th
from pathlib import Path
from time import time
from utils.parse_params import parse_input_paths

def find_min_and_max(tensor: th.Tensor) -> tuple:
    """
    Find the minimum and maximum values in a PyTorch tensor.

    Args:
        tensor (th.Tensor): The input tensor.

    Returns:
        tuple: A tuple containing the minimum and maximum values.
    """
    
    nan_mask = th.isnan(tensor)
    
    if nan_mask.all():
        return th.inf.item(), -th.inf.item()
    
    non_nan_tensor = tensor[~nan_mask]

    min_val = th.min(non_nan_tensor)
    max_val = th.max(non_nan_tensor)

    return min_val.item(), max_val.item()

def main():
    start_time = time()
    params, paths = parse_input_paths()
    
    dataset_kind = str(params["dataset"]["dataset_kind"])
    processed_data_dir = Path(paths[dataset_kind]["processed_data_dir"])
    
    if not processed_data_dir.exists():
        raise FileNotFoundError(f"Processed data directory {processed_data_dir} does not exist.")
    
    files_paths = list(processed_data_dir.glob("[0-9][0-9][0-9][0-9]_[0-9][0-9].pt"))
    if not files_paths:
        raise FileNotFoundError(f"No .pt files found in {processed_data_dir}.", flush=True)
    
    print(f"Found {len(files_paths)} files in {processed_data_dir}.")
    
    min_val, max_val = float('inf'), float('-inf')
    start_row = 1030
    start_col = 1280
    end_row = start_row + 168
    end_col = start_col + 144
    for file_path in files_paths:
        print(f"Processing file: {file_path}", flush=True)
        tensor = th.load(file_path)
        
        if not isinstance(tensor, th.Tensor):
            raise TypeError(f"File {file_path} does not contain a PyTorch tensor.")
        
        # Assuming the first channel is the one with the temperatures
        file_min, file_max = find_min_and_max(tensor[:, 0, start_row:end_row, start_col:end_col])
        min_val = min(min_val, file_min)
        max_val = max(max_val, file_max)
        print(f"File {file_path} - Min: {file_min}, Max: {file_max}\n", flush=True)
    
    th.save(th.tensor([min_val, max_val]), processed_data_dir / "min_max.pt")
    print(f"Minimum value: {min_val}, Maximum value: {max_val}", flush=True)
    print(f"Results saved to {processed_data_dir / 'min_max.pt'}", flush=True)
    print(f"Total time taken: {time() - start_time:.2f} seconds", flush=True)
    
    
if __name__ == "__main__":
    main()