"""
Example of how to load a netcdf file directly from acacia
"""

import xarray as xr
from s3fs import S3FileSystem, S3Map

# Create a text file with 3 lines
# Line 1: the endpoint url https://projects.pawsey.org.au/
# Line 2: access key
# Line 3: secret key
s3login = '/home/wedge/acacia_acc.txt'

def get_acacia_login(s3login):
    with open(s3login) as f:
        lines = f.readlines()

    url, key, secret = [ff.strip('\n') for ff in lines[0:3]]
    # Login to the s3 system
    s3 = S3FileSystem(client_kwargs={'endpoint_url':url},
            key=key,
            secret=secret)
    return s3


# s3file = 'uwaoceanprocesses-suntans/SUNTANS_125m/SurfaceGridded_2k_125m_nodff_20170215_swot_v2.nc'

def init_suntans_surface125(s3file, s3):
    # Make a map to the file location
    f = S3Map(s3file, s3=s3)
    return f

def load_suntans_surface125_xr():
    # Get the login details
    s3 = get_acacia_login(s3login)
    # Make a map to the file location
    f = init_suntans_surface125(s3file, s3)
    # Call xarray with the map object
    ds = xr.open_dataset(f)
    return ds

# # Call xarray with the map object
# ds = xr.open_dataset(f)