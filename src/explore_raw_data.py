import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# data_path = Path('data/IFREMER-GLOB-SST-L3-NRT-OBS_FULL_TIME_SERIE_202211/')
# data_path = Path('data/cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m/')
data_path = Path('data/raw/biochemistry/pft/')
# data_path = Path('data/GLOBAL_ANALYSISFORECAST_PHY_001_024/')
# data_path = Path('data/DIVAnd/')
# data_path = Path('data/EMODNET/')
# data_path = Path('data/ocldb1743602850/')
# data_path = Path('data/MULTIOBS_GLO_BIO_CARBON_SURFACE_MYNRT_015_008/')
# data_path = Path('data/raw/ensemble_physics/my/')

figs_folder = Path("figs/")

if not figs_folder.exists():
    raise FileNotFoundError(f"Directory {figs_folder} does not exist.")

# Check if the file exists
if not data_path.exists():
    print(f'Folder {data_path} does not exist')
    exit()

# Load the data
raw_data_paths = list(data_path.glob("*.nc"))
raw_data_path = raw_data_paths[0]
# raw_data_path = data_path / "Water_body_phosphate_subset.nc"
# raw_data_path = Path("data/raw/biochemistry/co2/2023_02.nc")
# raw_data_path = Path("data/raw/temperature/2023_01.nc")
raw_data_path = Path("data/raw/ensemble_physics/2023_01.nc")

if not raw_data_path.exists():
    print(f'File {raw_data_path} does not exist')
    exit()

data = xr.open_dataset(raw_data_path)

# Get the keys
print("Retrieving keys\n")
keys = list(data.keys())

print(f"Number of keys: {len(keys)}\n")

print(f"Keys: {keys}")

for key in keys:
    print(f"Key: {key}")
    print(f"Shape: {data[key].shape}")
    print()
    

# selected_key = "so_mean"

# images = data[selected_key].values[0, :, :, :]

# for c in range(images.shape[0]):
#     plt.imshow(images[c, :, :])
#     plt.colorbar()
#     plt.title(f"{selected_key} - {c}")
#     plt.savefig(f"figs/{selected_key}_{c}.png")
#     plt.close()

# shape = data[keys[0]].shape

# if len(shape) == 4: # Multy channel
#     for key in keys:  
#         for c in range(shape[1]):
#             plt.imshow(data[key].values[0, c, :, :])
#             plt.colorbar()
#             plt.title(f"{key} - {c}")
#             plt.savefig(f"figs/{key}_{c}.png")
#             plt.close()
# elif len(shape) == 3: # Single channel
#     print("Single channel")
#     for key in keys:
#         if key == 'crs':
#             continue
#         plt.imshow(data[key].values[0, :, :])
#         plt.colorbar()
#         plt.title(f"{key}")
#         plt.savefig(f"figs/{key}.png")
#         plt.close()
    
    
# sst_dtime = data['sst_dtime'].values[0]

# plt.imshow(sst_dtime)
# plt.show()

# print(data['crs'].values)

# n_lats = data["latitude"].shape[0]
# n_lons = data["longitude"].shape[0]
# n_days = data["time"].shape[0]

# print(f"Number of lats: {n_lats}")
# print(f"Number of lons: {n_lons}")
# print(f"Number of days: {n_days}")

# key_to_save = 'sea_surface_temperature'

# output_tensor = np.zeros((n_days, 1, n_lats, n_lons))


# output_tensor[:, 0, :, :] = np.array(data[key_to_save].values)


# np.savetxt("../sample.txt", output_tensor.reshape(-1, output_tensor.shape[-1]), delimiter=',')