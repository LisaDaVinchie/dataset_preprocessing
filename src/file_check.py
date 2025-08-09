from pathlib import Path
import zipfile

DATA_DIR = Path("./data/modis/raw/")
zip_files = sorted(DATA_DIR.glob("[0-9][0-9][0-9][0-9].zip"))
print(f"Found {len(zip_files)} zip files in {DATA_DIR}", flush=True)

for zip_path in zip_files:
    print(f"Checking {zip_path.name}", flush=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        if len(zf.namelist()) < 365 or len(zf.namelist()) > 366:
            raise ValueError(f"Invalid number of files in {zip_path.name}: {len(zf.namelist())}. Expected 365 or 366 files for daily data.")

    print(f"Finished checking {zip_path.name}\n", flush=True)