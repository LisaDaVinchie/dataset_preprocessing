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

def extract_sst_from_zip(zip_path: Path, lat_inds: np.ndarray, lon_inds: np.ndarray, output_file: Path, lat_subset: np.ndarray, lon_subset: np.ndarray):
    print(f"Processing {zip_path.name} ...\n", flush=True)
    lock = FileLock(str(output_file) + ".lock")
    daily_slices = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file in zf.namelist():
            if file.endswith('.nc'):
                with zf.open(file) as f:
                    with xr.open_dataset(BytesIO(f.read()), engine="h5netcdf") as ds:
                        subset = ds.isel(lat=lat_inds, lon=lon_inds)

                        # Optional
                        # Keep values where the temperature is in (0, 40) and the quality flag is between 1 and 3
                        # subset['sst'] = subset['sst'].where((subset['sst'] > 0) & (subset['sst'] < 40) & (subset['qual_sst'] < 4), np.nan)

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

    print(f"Extracted {len(daily_slices)} slices from {zip_path.name}", flush=True)
    
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

    with lock:
        if output_file.exists():
            print(f"Appending to existing file: {output_file}", flush=True)
            _append_to_netcdf(slices, output_file)
        else:
            print(f"Creating new file: {output_file}", flush=True)
            slices = slices.assign_coords(lat=("lat", lat_subset),
                                    lon=("lon", lon_subset))
            # slices = slices.transpose("lon", "lat", "time", ...)
            slices.to_netcdf(output_file, mode="w", format="NETCDF4",
                        unlimited_dims=["time"], engine="netcdf4")
    print(f"{zip_path.name} processed\n", flush=True)

def _append_to_netcdf(new_ds, output_file):
    with netCDF4.Dataset(output_file, "a") as nc:
        time_var = nc.variables['time']
        current_len = time_var.shape[0]
        
        new_len = new_ds.sizes['time']
        
        new_times = netCDF4.date2num(
            new_ds['time'].values.astype('datetime64[s]').tolist(),  # Ensure datetime64 input
            units=time_var.units,
            calendar=time_var.calendar
        )
        
        # Append time values
        time_var[current_len:current_len+new_len] = new_times
        print(f"Appending {new_len} new time values to {output_file}", flush=True)

        # Append each variable (adjust names as needed)
        for varname in ['sst', 'qual_sst', 'palette']:
            var = nc.variables[varname]
            # Assuming var dims order is (time, lat, lon)
            var[current_len:current_len+new_len, :, :] = new_ds[varname].values
    
if __name__ == "__main__":
    
    start_time = time.time()
    print("Program started", flush=True)

    # Area of interest
    lat_inds = np.arange(1050, 1050 + 168)
    lon_inds = np.arange(4600, 4600 + 144)

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
    print(f"Output file will be saved as: {OUTPUT_FILE}", flush=True)


    zip_files = sorted(DATA_DIR.glob("[0-9][0-9][0-9][0-9].zip"))
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

    # Prepare arguments for each zip file
    args = [
        (
            zip_file,
            lat_inds,
            lon_inds,
            OUTPUT_FILE,
            lat_subset,
            lon_subset
        )
        for zip_file in zip_files
    ]

    # This line automatically detects the number of CPUs
    n_processes = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    with Pool(processes=2) as pool:
        pool.starmap(extract_sst_from_zip, args)
        
    # Remove the lock file if it exists
    lock_file = str(OUTPUT_FILE) + ".lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)
        print(f"Removed lock file: {lock_file}", flush=True)

    print(f"Files processed in {time.time() - start_time:.2f} seconds\n", flush=True)