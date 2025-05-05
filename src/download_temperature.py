import copernicusmarine
import os
from utils.login import login_copernicus
import time

start_time = time.time()

login_copernicus()
print("\nLogin completed\n")

year = "2023"
month = "01"
day_start = "01"
day_end = "31"

dir_path = "data/raw/temperature/"
if not os.path.exists(dir_path):
    raise FileNotFoundError(f"Directory {dir_path} does not exist.")

copernicusmarine.subset(
    dataset_id="IFREMER-GLOB-SST-L3-NRT-OBS_FULL_TIME_SERIE",
    variables=["adjusted_sea_surface_temperature", "bias_to_reference_sst",
               "or_latitude", "or_longitude",
               "or_number_of_pixels", "quality_level",
               "satellite_zenith_angle", "sea_surface_temperature",
               "sea_surface_temperature_stddev", "solar_zenith_angle",
               "sses_bias", "sses_standard_deviation", "sst_dtime"],
    minimum_longitude=-179.97500610351562,
    maximum_longitude=179.97500610351562,
    minimum_latitude=-79.9749984741211,
    maximum_latitude=79.9749984741211,
    start_datetime=f"{year}-{month}-{day_start}T00:00:00",
    end_datetime=f"{year}-{month}-{day_end}T00:00:00",
    output_directory=dir_path,
    output_filename=f"{year}_{month}.nc"
)

print(f"\nDownload completed in {time.time() - start_time} seconds\n")