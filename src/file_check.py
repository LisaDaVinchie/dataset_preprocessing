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
            year = int(zip_path.stem)
            start_date = datetime.date(year, 1, 1)
            days_in_year = (datetime.date(year, 12, 31) - start_date).days + 1
            year_dates = [(start_date + datetime.timedelta(days=i)).strftime("%Y%m%d") for i in range(days_in_year)]
            available_dates = [file.split('.')[1] for file in zf.namelist()]
            for date in year_dates:
                if date not in available_dates:
                    print(f"Missing file for {date} in {zip_path.name}", flush=True)

    print(f"Finished checking {zip_path.name}\n", flush=True)