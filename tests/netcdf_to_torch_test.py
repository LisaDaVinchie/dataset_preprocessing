import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from netcdf_to_torch.ensemble_physics import NetcdfToTorch

class TestNetcdfToTorch(unittest.TestCase):
    def setUp(self):
        self.channels_to_keep = ["channel1", "channel2"]
        self.n_rows = 100
        self.n_cols = 200
        self.dataset_name = "test_dataset"
        self.year_position = [41, 45]
        self.month_position = [45, 47]
        self.day_position = [47, 49]
        # Create a sample params file
        self.params_content = {
            "dataset": {
                "channels_to_keep": self.channels_to_keep,
                "n_rows": self.n_rows,
                "n_cols": self.n_cols,
                "dataset_name": self.dataset_name
            },
            "test_dataset": {
                "year_position": self.year_position,
                "month_position": self.month_position,
                "day_position": self.day_position
            }
        }
        
        self.temp_json = NamedTemporaryFile(delete=False, suffix=".json", mode='w')
        json.dump(self.params_content, self.temp_json)
        self.temp_json.close()
        self.params_path = Path(self.temp_json.name).resolve()
        
        self.netcdf2torch = NetcdfToTorch(
            raw_data_dir=Path("data/raw/"),
            processed_data_dir=Path("data/processed/ensemble_physics/"),
            processed_data_ext=".pt",
            params_path=self.params_path
        )
        
    def tearDown(self):
        Path(self.temp_json.name).unlink()
        
    def test_load_params(self):
        # Test if the parameters are loaded correctly
        self.assertEqual(self.netcdf2torch.keys_to_keep, self.channels_to_keep)
        self.assertEqual(self.netcdf2torch.n_rows, self.n_rows)
        self.assertEqual(self.netcdf2torch.n_cols, self.n_cols)
        self.assertEqual(self.netcdf2torch.year_position, self.year_position)
        self.assertEqual(self.netcdf2torch.month_position, self.month_position)
        self.assertEqual(self.netcdf2torch.day_position, self.day_position)
        
    def test_generate_processed_data_path(self):
        
        sample_file_path = Path("/Users/lisadavinchie/Documents/University/Thesis/Dataset_creation/data/raw/ensemble_physics/my/cmems_mod_glo_phy-mnstd_my_0.25deg_P1D-m-20230101.nc")
        
        expected_processed_data_path = Path("data/processed/ensemble_physics/2023_01_01.pt")
        
        processed_data_path = self.netcdf2torch.generate_processed_data_path(sample_file_path)
        
        self.assertEqual(processed_data_path, expected_processed_data_path)

if __name__ == '__main__':
    unittest.main()