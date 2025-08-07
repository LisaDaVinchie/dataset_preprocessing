import zipfile
import xarray as xr
import numpy as np
from io import BytesIO
from pathlib import Path
from datetime import datetime
import time
start_time = time.time()


LON_RANGE = (-7, 37)
LAT_RANGE = (30, 46)
# Area of interest
lat_inds = np.arange(1056, 1439 + 1)
lon_inds = np.arange(4152, 5207 + 1)

print(f"Lat indices: {len(lat_inds)}, lon indices: {len(lon_inds)}")

# Paths
DATA_DIR = Path("./data/modis/raw/")
OUTPUT_DIR = Path("./data/modis/processed/")
OUTPUT_DIR.mkdir(exist_ok=True)

i = 1
OUTPUT_FILE = Path(f"./data/modis/processed/dataset_{i}.nc")
while OUTPUT_FILE.exists():
    i += 1
    OUTPUT_FILE = Path(f"./data/modis/processed/dataset_{i}.nc")


zip_files = sorted(DATA_DIR.glob("[0-9][0-9][0-9][0-9].zip"))
print(f"Found {len(zip_files)} zip files in {DATA_DIR}")

def extract_sst_from_zip(zip_path: Path):
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
    return daily_slices

# Process each zip
sst_list = []
for zip_path in zip_files:
    print(f"Processing {zip_path.name} ...")
    sst_list.extend(extract_sst_from_zip(zip_path))
    print(f"{zip_path.name} processed")

print(f"Files processed in {time.time() - start_time:.2f} seconds\n")

t2 = time.time()
# Combine all slices
combined = xr.concat(sst_list, dim="time")

print(f"Files concatenated in {time.time() - t2:.2f} seconds\n")

with zipfile.ZipFile(zip_files[0], 'r') as zf:
    for file in zf.namelist():
        if file.endswith('.nc'):
            with zf.open(file) as f:
                with xr.open_dataset(BytesIO(f.read()), engine="h5netcdf") as ds:
                    lat_subset = ds.lat.isel(lat=lat_inds).values
                    lon_subset = ds.lon.isel(lon=lon_inds).values
                    # lat_subset = ds.lat.sel(lat=slice(46, 30)).values
                    # lon_subset = ds.lon.sel(lon=slice(-7, 37)).values
            break  # just use the first file

# After combining
combined = combined.assign_coords(lat=("lat", lat_subset),
                                    lon=("lon", lon_subset))

# Reorder dimensions to (lon, lat, time)
reordered = combined.transpose("lon", "lat", "time", ...)

print("Indexes reordered\n")

# Save to NetCDF
combined.to_netcdf(OUTPUT_FILE, format="NETCDF4", engine="netcdf4")
print(f"✅ Saved with shape {combined['sst'].shape} to {OUTPUT_FILE} in {time.time() - start_time:.2f} seconds\n")