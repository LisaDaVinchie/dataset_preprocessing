from pathlib import Path
import zipfile
import datetime

DATA_DIR = Path("./data/modis/raw/")
zip_files = sorted(DATA_DIR.glob("[0-9][0-9][0-9][0-9].zip"))
print(f"Found {len(zip_files)} zip files in {DATA_DIR}", flush=True)

for zip_path in zip_files:
    print(f"Checking {zip_path.name}", flush=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        if len(zf.namelist()) < 365 or len(zf.namelist()) > 366:
            print(f"Invalid number of files in {zip_path.name}: {len(zf.namelist())}. Expected 365 or 366 files for daily data.", flush= True)
            for i in range(1, 13):
                if f"TERRA_MODIS.{i:02d}" not in zf.namelist():
                    print(f"Missing file for month {i:02d} in {zip_path.name}", flush=True)
                # Determine the number of days in the month for the current year
                year = int(zip_path.stem)
                n_days = (datetime.date(year if i < 12 else year + 1, i % 12 + 1, 1) - datetime.date(year, i, 1)).days
                for j in range(1, n_days + 1):
                    if f"TERRA_MODIS.{i:02d}.{j:02d}" not in zf.namelist():
                        print(f"Missing file for day {j:02d} of month {i:02d} in {zip_path.name}", flush=True)

    print(f"Finished checking {zip_path.name}\n", flush=True)