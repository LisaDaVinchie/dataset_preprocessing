import zipfile
import xarray as xr
import numpy as np
from io import BytesIO
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
import os
import time
import netCDF4
from filelock import FileLock
start_time = time.time()
print("Program started", flush=True)


# LON_RANGE = (-7, 37)
# LAT_RANGE = (30, 46)
# Area of interest
lat_inds = np.arange(1050, 1050 + 168 + 1)
lon_inds = np.arange(4600, 4600 + 144 + 1)

print(f"Lat indices: {len(lat_inds)}, lon indices: {len(lon_inds)}", flush=True)

# Paths
DATA_DIR = Path("./data/modis/raw/")
OUTPUT_DIR = Path("./data/modis/processed/")
OUTPUT_DIR.mkdir(exist_ok=True)
print("Output directory created:", OUTPUT_DIR, flush=True)

i = 1
OUTPUT_FILE = Path(f"./data/modis/processed/dataset_{i}.nc")
while OUTPUT_FILE.exists():
    i += 1
    OUTPUT_FILE = Path(f"./data/modis/processed/dataset_{i}.nc")
lock = FileLock(str(OUTPUT_FILE) + ".lock")
print(f"Output file will be saved as: {OUTPUT_FILE}", flush=True)


zip_files = sorted(DATA_DIR.glob("[0-9][0-9][0-9][0-9].zip"))[0:3]
print(f"Found {len(zip_files)} zip files in {DATA_DIR}", flush=True)

with zipfile.ZipFile(zip_files[0], 'r') as zf:
    for file in zf.namelist():
        if file.endswith('.nc'):
            with zf.open(file) as f:
                with xr.open_dataset(BytesIO(f.read()), engine="h5netcdf") as ds:
                    lat_subset = ds.lat.isel(lat=lat_inds).values
                    lon_subset = ds.lon.isel(lon=lon_inds).values
            break  # just use the first file
print("Coordinates extracted from the first file\n", flush=True)

# encoding = {
#     'sst': {
#         'dtype': 'float32',
#         'zlib': True
#     },
#     'qual_sst': {
#         'dtype': 'float32',
#         'zlib': True
#     },
#     'time': {
#         'dtype': 'int64',
#         'zlib': False
#     },
#     'palette': {
#         'zlib': False
#     }
# }

def append_to_netcdf(output_file, new_ds):
    with netCDF4.Dataset(output_file, "a") as nc:
        time_var = nc.variables['time']
        current_len = time_var.shape[0]
        
        new_len = new_ds.dims['time']
        
        new_times = netCDF4.date2num(
            new_ds['time'].values.astype('datetime64[s]').tolist(),  # Ensure datetime64 input
            units=time_var.units,
            calendar=time_var.calendar
        )
        
        # Append time values
        time_var[current_len:current_len+new_len] = new_times
        
        # Append each variable (adjust names as needed)
        for varname in ['sst', 'qual_sst', 'palette']:
            var = nc.variables[varname]
            # Assuming var dims order is (time, lat, lon)
            data = new_ds[varname].values
            var[current_len:current_len+new_len, :, :] = data


def extract_sst_from_zip(zip_path: Path):
    print(f"Processing {zip_path.name} ...\n", flush=True)
    daily_slices = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file in zf.namelist():
            if file.endswith('.nc'):
                with zf.open(file) as f:
                    with xr.open_dataset(BytesIO(f.read()), engine="h5netcdf") as ds:
                        subset = ds.isel(lat=lat_inds, lon=lon_inds)
                        
                        # Parse date from filename
                        date_str = file.split('.')[1]  # TERRA_MODIS.YYYYMMDD...
                        date = np.datetime64(datetime.strptime(date_str, "%Y%m%d"))

                        # Add time dimension
                        subset = subset.expand_dims(dim="time")
                        subset = subset.assign_coords(time=("time", [date]))

                        # subset = subset.reset_coords(names=["lat", "lon"], drop=True)
                        subset = subset.drop_vars(["lat", "lon"])

                        daily_slices.append(subset)
                        
    slices = xr.concat(daily_slices, dim="time")
    slices = xr.Dataset(
        {
            "sst": slices["sst"],
            "qual_sst": slices["qual_sst"],
            "palette": slices["palette"]
        },
        coords={
            "time": slices["time"],
            "lat": slices["lat"],
            "lon": slices["lon"]
        }
    )
    print(f"Extracted {len(daily_slices)} slices from {zip_path.name}", flush=True)
    
    with lock:
        if OUTPUT_FILE.exists():
            print(f"Appending to existing file: {OUTPUT_FILE}", flush=True)
            slices.to_netcdf(OUTPUT_FILE, mode="a", format="NETCDF4",
                        unlimited_dims=["time"], engine="netcdf4")
        else:
            print(f"Creating new file: {OUTPUT_FILE}", flush=True)
            slices = slices.assign_coords(lat=("lat", lat_subset),
                                    lon=("lon", lon_subset))
            slices = xr.Dataset(
                {
                    "sst": slices["sst"],
                    "qual_sst": slices["qual_sst"],
                    "palette": slices["palette"]
                },
                coords={
                    "time": slices["time"],
                    "lat": slices["lat"],
                    "lon": slices["lon"]
                }
            )
            slices = slices.transpose("lon", "lat", "time", ...)
            slices.to_netcdf(OUTPUT_FILE, mode="w", format="NETCDF4",
                        unlimited_dims=["time"], engine="netcdf4")
    print(f"{zip_path.name} processed\n", flush=True)

# This line automatically detects the number of CPUs
n_processes = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))

with Pool(processes=n_processes) as pool:
    pool.map(extract_sst_from_zip, zip_files)

print(f"Files processed in {time.time() - start_time:.2f} seconds\n", flush=True)