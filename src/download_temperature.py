from utils.login import login_copernicus
from utils.download_dataset import CopernicusMarineDownloader
import time
import argparse
from pathlib import Path
import json

start_time = time.time()

parser = argparse.ArgumentParser(description="Download temperature data from Copernicus Marine Service.")
parser.add_argument("--paths", type=str, required=True, help="Path to the json containing the paths.")
args = parser.parse_args()

paths_file = Path(args.paths)
with open(paths_file, "r") as f:
    paths = json.load(f)
    
raw_data_dir = Path(paths["temperature"]["raw_data_dir"])
if not raw_data_dir.exists():
    raise FileNotFoundError(f"Directory {raw_data_dir} does not exist.")

login_copernicus()
print("\nLogin completed\n")

year = "2023"
month = "01"
day_start = "01"
day_end = "31"

dwl = CopernicusMarineDownloader(
    longitude_range=[-179.97500610351562, 179.97500610351562],
    latitude_range=[-79.9749984741211, 79.9749984741211],
    datetime_range=[f"{year}-{month}-{day_start}T00:00:00", f"{year}-{month}-{day_end}T00:00:00"],
    depth_range=[0.4940253794193268, 0.4940253794193268]
)
print("\nCopernicus Marine Downloader initialized\n")
# Download the data

dwl.download(
    output_filename=f"{year}_{month}.nc",
    dataset_id="IFREMER-GLOB-SST-L3-NRT-OBS_FULL_TIME_SERIE",
    output_directory=raw_data_dir,
    variables=["adjusted_sea_surface_temperature", "bias_to_reference_sst",
               "or_latitude", "or_longitude",
               "or_number_of_pixels", "quality_level",
               "satellite_zenith_angle", "sea_surface_temperature",
               "sea_surface_temperature_stddev", "solar_zenith_angle",
               "sses_bias", "sses_standard_deviation", "sst_dtime"]
)

print(f"\nDownload completed in {time.time() - start_time} seconds\n")