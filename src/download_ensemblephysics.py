import copernicusmarine
import os
from utils.login import login_copernicus
import time
import argparse
from pathlib import Path
import json

start_time = time.time()

parser = argparse.ArgumentParser(description="Download ensemble physics data from Copernicus Marine Service.")
parser.add_argument("--paths", type=str, required=True, help="Path to the json containing the paths.")
args = parser.parse_args()

paths_file = Path(args.paths)
with open(paths_file, "r") as f:
    paths = json.load(f)
    
raw_data_dir = Path(paths["ensemble_physics"]["raw_data_dir"])
if not raw_data_dir.exists():
    raise FileNotFoundError(f"Directory {raw_data_dir} does not exist.")

login_copernicus()
print("\nLogin completed\n")

year = "2023"
month = "01"
day_start = "01"
day_end = "31"

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy-all_my_0.25deg_P1D-m",
    variables=["mlotst_cglo", "mlotst_glor", "mlotst_oras", "siconc_cglo", "siconc_glor", "siconc_oras", "sithick_cglo", "sithick_glor", "sithick_oras", "so_cglo", "so_glor", "so_oras", "thetao_cglo", "thetao_glor", "thetao_oras", "uo_cglo", "uo_glor", "uo_oras", "vo_cglo", "vo_glor", "vo_oras", "zos_cglo", "zos_glor", "zos_oras"],
    minimum_longitude=-180,
    maximum_longitude=179.75,
    minimum_latitude=-80,
    maximum_latitude=90,
    start_datetime=f"{year}-{month}-{day_start}T00:00:00",
    end_datetime=f"{year}-{month}-{day_end}T00:00:00",
    output_directory=raw_data_dir,
    output_filename=f"{year}_{month}.nc"
)

print(f"\nDownload completed in {time.time() - start_time} seconds\n")