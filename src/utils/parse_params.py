import json
import argparse
from pathlib import Path

def parse_input_paths() -> tuple:
    """Parse input paths from command line arguments.

    Returns:
        tuple: A tuple containing the parameters and paths loaded from JSON files.
    """
    parser = argparse.ArgumentParser(description='Convert NetCDF to Torch')
    parser.add_argument('--paths', type=Path, help='File containing paths', required=True)
    parser.add_argument('--params', type=Path, help='Output directory', required=True)

    args = parser.parse_args()
    paths_path = Path(args.paths)
    params_path = Path(args.params)
    
    if not paths_path.exists():
        raise FileNotFoundError(f"File {paths_path} does not exist.")
    if not params_path.exists():
        raise FileNotFoundError(f"File {params_path} does not exist.")

    with open(paths_path, "r") as f:
        paths = json.load(f)
    
    with open(params_path, "r") as f:
        params = json.load(f)
    return params, paths