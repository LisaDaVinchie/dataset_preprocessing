import copernicusmarine
import os
import time
from utils.login import login_copernicus

start_time = time.time()

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
    # "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m"
    output_directories[name] = f"data/raw/biochemistry/{name}"
    
    if not os.path.exists(output_directories[name]):
        raise FileNotFoundError(f"Directory {output_directories[name]} does not exist.")  
print("\nDirectories checked\n")

variables = {
    "bio": ["nppv", "o2"],
    "car": ["dissic", "ph", "talk"],
    "co2": ["spco2"],
    "nut": ["fe", "no3", "po4", "si"],
    "pft": ["chl", "phyc"],
    "plankton": ["zooc"]
}

login_copernicus()
print("\nLogin completed\n")

# Download biochemistry data from Copernicus Marine Service
for name in dataset_name:
    print(f"\nDownloading {name} data from Copernicus Marine Service\n")
    copernicusmarine.subset(
        dataset_id=dataset_id[name],
        variables=variables[name],
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        minimum_depth=minimum_depth,
        maximum_depth=maximum_depth,
        output_directory=output_directories[name],
        output_filename=output_filename
    )
    print(f"Downloaded {name} data from Copernicus Marine Service\n\n")
    
print(f"All data downloaded in {time.time() - start_time} seconds\n")