from utils.login import login_copernicus
from utils.download_dataset import CopernicusMarineDownloader
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

dwl = CopernicusMarineDownloader(
    longitude_range=[-180, 179.75],
    latitude_range=[-80, 90],
    datetime_range=[f"{year}-{month}-{day_start}T00:00:00", f"{year}-{month}-{day_end}T00:00:00"],
    depth_range=[0.5057600140571594, 0.5057600140571594]
)

print("Constructed downloader\n")

dwl.download(
    output_filename=f"{year}_{month}.nc",
    dataset_id="cmems_mod_glo_phy-mnstd_my_0.25deg_P1D-m",
    output_directory=raw_data_dir,
    variables=["mlotst_mean", "mlotst_std", "siconc_mean", "siconc_std", "sithick_mean", "sithick_std", "so_mean", "so_std", "thetao_mean", "thetao_std", "uo_mean", "uo_std", "vo_mean", "vo_std", "zos_mean", "zos_std"]
)

print(f"\nDownload completed in {time.time() - start_time} seconds\n")