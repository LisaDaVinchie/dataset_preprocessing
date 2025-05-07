import copernicusmarine

class CopernicusMarineDownloader:
    def __init__(self, longitude_range: list, latitude_range: list, depth_range: list):
        """Initialize the Copernicus Marine Downloader.

        Args:
            longitude_range (list): list of two floats representing the minimum and maximum longitude
            latitude_range (list): list of two floats representing the minimum and maximum latitude
            depth_range (list): list of two floats representing the minimum and maximum depth
        """
        self.minimum_longitude = longitude_range[0]
        self.maximum_longitude = longitude_range[1]
        self.minimum_latitude = latitude_range[0]
        self.maximum_latitude = latitude_range[1]
        self.minimum_depth = depth_range[0]
        self.maximum_depth = depth_range[1]

    def download(self, output_filename: str, dataset_id: str, output_directory: str, variables: list, datetime_range: dict):
        """Download data from Copernicus Marine Service.

        Args:
            output_filename (str): file name of the saved dataset
            dataset_id (str): dataset id
            output_directory (str): directory to save the dataset
            variables (list): list of variables of the dataset to download
            datetime_range (dict): dictionary containing the start and end datetime for the download
        """
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=self.minimum_longitude,
            maximum_longitude=self.maximum_longitude,
            minimum_latitude=self.minimum_latitude,
            maximum_latitude=self.maximum_latitude,
            start_datetime=datetime_range[0],
            end_datetime=datetime_range[1],
            minimum_depth=self.minimum_depth,
            maximum_depth=self.maximum_depth,
            output_directory=output_directory,
            output_filename=output_filename
        )