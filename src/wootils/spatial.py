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


def ds_2_utm(ds, ds_vars=None, ll_coords=['longitude', 'latitude'], mesh=True, convert_coords=None):
    if (convert_coords is not None) & (mesh is True):
        try:
            ds = ds.transpose(ll_coords[0], ll_coords[1], convert_coords)
        except ValueError:
            raise ValueError(f"Dataset dimensions {ds.dims} do not match the provided coordinates {ll_coords} and {convert_coords}.")

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

    # Optionally convert coordinates
    if convert_coords is None:
        return ds    
    else:
        # Build a new dataset with the converted coordinates
        ds_converted = xr.Dataset()
        for var in ds.data_vars:
            if var not in ll_coords:
                if mesh:
                    if var not in ['easting', 'northing']:
                        ds_converted[var] = xr.DataArray(
                            ds[var].values, dims=['easting', 'northing'] + [convert_coords],
                            coords={'easting': ds['easting'].values.mean(axis=0), 'northing': ds['northing'].values.mean(axis=1), convert_coords: ds[convert_coords]})
                else:
                    ds_converted[var] = ds[var]
    
        if not mesh:
            ds_converted['easting'] = ds['easting']
            ds_converted['northing'] = ds['northing']
        return ds_converted