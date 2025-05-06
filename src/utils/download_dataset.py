import copernicusmarine

class CopernicusMarineDownloader:
    def __init__(self, longitude_range: list, latitude_range: list, datetime_range: list, depth_range: list):
        """Initialize the Copernicus Marine Downloader.

        Args:
            longitude_range (list): list of two floats representing the minimum and maximum longitude
            latitude_range (list): list of two floats representing the minimum and maximum latitude
            datetime_range (list): list of two strings representing the start and end datetime, format "YYYY-MM-DDT00:00:00"
            depth_range (list): list of two floats representing the minimum and maximum depth
        """
        self.minimum_longitude = longitude_range[0]
        self.maximum_longitude = longitude_range[1]
        self.minimum_latitude = latitude_range[0]
        self.maximum_latitude = latitude_range[1]
        self.start_datetime = datetime_range[0]
        self.end_datetime = datetime_range[1]
        self.minimum_depth = depth_range[0]
        self.maximum_depth = depth_range[1]

    def download(self, output_filename: str, dataset_id: str, output_directory: str, variables: list):
        """Download data from Copernicus Marine Service.

        Args:
            output_filename (str): file name of the saved dataset
            dataset_id (str): dataset id
            output_directory (str): directory to save the dataset
            variables (list): list of variables of the dataset to download
        """
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=self.minimum_longitude,
            maximum_longitude=self.maximum_longitude,
            minimum_latitude=self.minimum_latitude,
            maximum_latitude=self.maximum_latitude,
            start_datetime=self.start_datetime,
            end_datetime=self.end_datetime,
            minimum_depth=self.minimum_depth,
            maximum_depth=self.maximum_depth,
            output_directory=output_directory,
            output_filename=output_filename
        )