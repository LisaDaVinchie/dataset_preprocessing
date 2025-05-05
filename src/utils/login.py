import json
from copernicusmarine import login
from pathlib import Path

def login_copernicus():
    login_path = Path("src/login.json")
    if not login_path.exists():
        raise FileNotFoundError(f"Login file {login_path} does not exist.")
    
    with open(login_path, "r") as f:
        credentials = json.load(f)
    
    username = credentials["username"]
    password = credentials["password"]

    # Login
    login(username=username, password=password, check_credentials_valid=True, force_overwrite=False, configuration_file_directory="data")