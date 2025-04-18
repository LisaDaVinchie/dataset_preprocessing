import xarray as xr

# url = "http://opendap.oceanbrowser.net/thredds/dodsC/data/emodnet-projects/Phase-2/Baltic%20Sea/Summer%20(June-August)%20-%2010-years%20running%20averages/Water_body_phosphate.4Danl.nc"
url = "http://opendap.oceanbrowser.net/thredds/dodsC/data/emodnet-projects/Phase-2/Baltic%20Sea/Summer%20(June-August)%20-%2010-years%20running%20averages/Water_body_phosphate.4Danl.nc?lon[0:1:215],lat[0:1:129],depth[0:1:20]"
ds = xr.open_dataset(url)

# Save to NetCDF
ds.to_netcdf("Water_body_phosphate_subset.nc")
print("File saved successfully!")