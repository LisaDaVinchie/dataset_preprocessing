from utils.login import login_copernicus
from utils.parse_params import parse_input_paths
from utils.download_dataset import CopernicusMarineDownloader
import time
from pathlib import Path
import calendar

def main():
    start_time = time.time()
    
    params, paths = parse_input_paths()
    print("Input paths parsed successfully\n")

    dataset_name = params["dataset"]["dataset_name"]
        
    raw_data_dir = Path(paths[dataset_name]["raw_data_dir"])
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Directory {raw_data_dir} does not exist.")

    longitude_range = list(params[dataset_name]["longitude_range"])
    latitude_range = list(params[dataset_name]["latitude_range"])
    depth_range = list(params[dataset_name]["depth_range"])
    channels = list(params[dataset_name]["channels"])
    dataset_id = str(params[dataset_name]["dataset_id"])
    year_range = list(params["dataset"]["year_range"])
    month_range = list(params["dataset"]["month_range"])

    datetime_range = get_datetime_range(year_range, month_range) 
    
    print("datetime_range:", datetime_range)       

    login_copernicus()
    print("\nLogin completed\n")

    dwl = CopernicusMarineDownloader(
        longitude_range=longitude_range,
        latitude_range=latitude_range,
        depth_range=depth_range
    )
    print("\nCopernicus Marine Downloader initialized\n")
    # Download the data

    for year in year_range:
        for month in month_range:
            print(f"\nDownloading data for {year}-{month}...\n")
            dwl.download(
                output_filename=f"{year}_{month}.nc",
                dataset_id=dataset_id,
                output_directory=raw_data_dir,
                variables=channels,
                datetime_range=datetime_range[(year, month)]
            )
            print(f"\nDownload completed for {year}-{month} in {time.time() - start_time} seconds\n")

    print(f"\nDownload completed in {time.time() - start_time} seconds\n")

def get_datetime_range(year_range: list, month_range: list) -> dict:
    # Check if the month range is valid
    if len(month_range) != 2 or month_range[0] > month_range[1]:
        raise ValueError("Invalid month range. It should be a list of two integers [start_month, end_month].")
    # Check if the year range is valid
    if len(year_range) != 2 or year_range[0] > year_range[1]:
        raise ValueError("Invalid year range. It should be a list of two integers [start_year, end_year].")

    # Get day range for each month
    datetime_range = {}
    for year in range(year_range[0], year_range[1] + 1):
        for month in range(month_range[0], month_range[1] + 1):
            datetime_start = f"{year}-{month}-01T00:00:00"
            days_in_month = calendar.monthrange(year, month)[1]
            
            datetime_end = f"{year}-{month}-{days_in_month}T00:00:00"
            datetime_range[(year, month)] = [datetime_start, datetime_end]
    return datetime_range

if __name__ == "__main__":
    main()