import os
import time
import argparse
from pathlib import Path
import json
from utils.login import login_copernicus
from utils.download_dataset import CopernicusMarineDownloader

start_time = time.time()

parser = argparse.ArgumentParser(description="Download biochemistry data from Copernicus Marine Service.")
parser.add_argument("--paths", type=str, required=True, help="Path to the json containing the paths.")
args = parser.parse_args()

paths_file = Path(args.paths)
with open(paths_file, "r") as f:
    paths = json.load(f)
    
raw_data_dir = Path(paths["biochemistry"]["raw_data_dir"])
if not raw_data_dir.exists():
    raise FileNotFoundError(f"Directory {raw_data_dir} does not exist.")

minimum_longitude = -180
maximum_longitude = 179.75
minimum_latitude = -80
maximum_latitude = 90

year = "2023"
month = "01"
day_start = "01"
day_end = "31"

start_datetime = f"{year}-{month}-{day_start}T00:00:00"
end_datetime = f"{year}-{month}-{day_end}T00:00:00"
output_filename = f"{year}_{month}.nc"

minimum_depth = 0.4940253794193268
maximum_depth = 0.4940253794193268

dataset_name = ["bio", "car", "co2", "nut", "pft"]

dataset_id = {}
output_directories = {}
for name in dataset_name:
    dataset_id[name] = f"cmems_mod_glo_bgc-{name}_anfc_0.25deg_P1D-m"
    output_directories[name] = raw_data_dir / f"{name}/"
    
    if not os.path.exists(output_directories[name]):
        raise FileNotFoundError(f"Directory {output_directories[name]} does not exist.")  
print("\nDirectories checked\n")

variables = {
    "bio": ["nppv", "o2"],
    "car": ["dissic", "ph", "talk"],
    "co2": ["spco2"],
    "nut": ["fe", "no3", "po4", "si"],
    "pft": ["chl", "phyc"]
}

login_copernicus()
print("\nLogin completed\n")

dwl = CopernicusMarineDownloader(
    longitude_range=[minimum_longitude, maximum_longitude],
    latitude_range=[minimum_latitude, maximum_latitude],
    datetime_range=[start_datetime, end_datetime],
    depth_range=[minimum_depth, maximum_depth]
)
print("\nCopernicus Marine Downloader initialized\n")

for name in dataset_name:
    print(f"\nDownloading {name} data from Copernicus Marine Service\n")
    dwl.download(
        output_filename=output_filename,
        dataset_id=dataset_id[name],
        output_directory=output_directories[name],
        variables=variables[name]
    )
    print(f"Downloaded {name} data from Copernicus Marine Service\n\n")
    
print(f"All data downloaded in {time.time() - start_time} seconds\n")