import numpy as np
import pyproj as pp
import xarray as xr


def calc_geodistance(longitude, latitude, ellps='WGS84', **ppkwargs):
    p = pp.Geod(ellps=ellps, **ppkwargs)
    dist = np.full(len(longitude), np.nan)
    dist[1:] = p.inv(longitude[:-1], latitude[:-1], longitude[1:], latitude[1:])[2]
    return dist



def calc_east_north(lon, lat, proj='utm', zone=51, ellps='WGS84', south=True,
                    inverse=False, errcheck=False, radians=False, *ppkwargs):
    p = pp.Proj(proj=proj, zone=zone, ellps=ellps, south=south, *ppkwargs)
    return p(lon, lat, inverse=inverse, errcheck=errcheck, radians=radians)


def ds_2_utm(ds, ds_vars=None, ll_coords=['longitude', 'latitude'], mesh=True):

    if ds_vars is None:
        # Get all variables with ll_coords as dimensions
        ds_vars = [v for v in ds.data_vars if all(d in ds[v].dims for d in ll_coords)]

    # Grid the lat-lon positions
    if mesh:
        lon_gg, lat_gg = np.meshgrid(ds[ll_coords[0]].values, ds[ll_coords[1]].values)
    else:
        lon_gg, lat_gg = ds[ll_coords[0]].values, ds[ll_coords[1]].values
    
    # Get the UTM coordinates
    east, north = calc_east_north(lon_gg, lat_gg)

    # Add the UTM coordinates to the dataset
    if mesh:
        ds['easting'] = xr.DataArray(east, dims=np.flip(ll_coords), coords={ll_coords[0]: ds[ll_coords[0]], ll_coords[1]: ds[ll_coords[1]]})
        ds['northing'] = xr.DataArray(north, dims=np.flip(ll_coords), coords={ll_coords[0]: ds[ll_coords[0]], ll_coords[1]: ds[ll_coords[1]]})
    else:
        ds['easting'] = xr.DataArray(east, dims=ds.dims)
        ds['northing'] = xr.DataArray(north, dims=ds.dims)
    # for v in ds_vars:
    #     ds[v] = ds[v].assign_coords(easting=('longitude', east), northing=('latitude', north))

    return ds