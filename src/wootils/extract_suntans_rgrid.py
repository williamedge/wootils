import pyproj as pp
import numpy as np
import xarray as xr
# from tqdm import tqdm
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation


class ubox():
    '''Class to store a bounding box for a SUNTANS unstructured grid model output.'''
    def __init__(self, xlims, ylims, tlims):
        self.xlims = xlims
        self.ylims = ylims
        self.tlims = tlims


    def bounds(self, ds, type='node'):
        '''Call the unstructured grid bounds function.'''
        ix, iy, it = ugrid_bounds(ds, self.xlims, self.ylims, self.tlims, type=type)
        self.ix = ix
        self.iy = iy
        self.it = it
        self.type = type


    def cutin(self, cutval):
        '''Call the box_cutin function.'''
        cutbox = box_cutin(self, cutval=cutval)
        return cutbox


    def plot(self, ax, **kwargs):
        '''Plot the bounding box.'''
        rect = plt.Rectangle((self.xlims[0], self.ylims[0]), np.diff(self.xlims)[0], np.diff(self.ylims)[0],\
                            **kwargs)
        ax.add_patch(rect)


    def shortside(self, north=False):
        '''Call the get_shortside function.'''
        self.boxlen = get_shortside(self.xlims, self.ylims, north=north)
        return self.boxlen


    def squaregrid(self, side_len_km, resolution_km, corner_utm, start_corner=[0,0], round_km=0):
        '''Call the make_squaregrid function.'''
        self.xgrid, self.ygrid = make_squaregrid(side_len_km, resolution_km, corner_utm, start_corner=start_corner, round_km=round_km)
        return self.xgrid, self.ygrid


def ugrid_bounds(ds, xlims, ylims, tlims, type='node'):
    '''Get the lon, lat and time indices for a given bounding box and time range from a SUNTANS unstructured grid model output.'''

    if type == 'node':
        xvar, yvar = 'xp', 'yp'
    elif type == 'face':
        xvar, yvar = 'xv', 'yv'
    else:
        raise ValueError('type must be node or face')

    ix = (ds[xvar] >= xlims[0]) & (ds[xvar] <= xlims[1])
    iy = (ds[yvar] >= ylims[0]) & (ds[yvar] <= ylims[1])
    it = ds['time'].sel(time=slice(tlims[0], tlims[1]))
    return ix, iy, it


def box_cutin(bigbox, cutval):
    '''Spatially cut in a smaller box from the larger box by the cutval amount (in degrees).'''
    xlims = [bigbox.xlims[0] + cutval, bigbox.xlims[1] - cutval]
    ylims = [bigbox.ylims[0] + cutval, bigbox.ylims[1] - cutval]
    tlims = bigbox.tlims
    return ubox(xlims, ylims, tlims)


def calc_east_north(lon, lat, proj='utm', zone=51, ellps='WGS84', south=True, **ppkwargs):
    p = pp.Proj(proj=proj, zone=zone, ellps=ellps, south=south, **ppkwargs)
    return p(lon, lat)


def get_shortside(xlims, ylims, north=False):
    '''Get the shortest side of a box in km.'''
    if not north:
        east_bounds,_ = calc_east_north(xlims, [ylims[0], ylims[0]])
    elif north:
        east_bounds,_ = calc_east_north(xlims, [ylims[1], ylims[1]])
    return np.diff(east_bounds)[0]/1000


def make_squaregrid(side_len_km, resolution_km, corner_utm, start_corner=[0,0], round_km=0):
    '''Make a regular grid of points in km.'''
    if start_corner[0] == 1:
        x_spacing = -resolution_km
        x_len = -side_len_km
    else:
        x_spacing = resolution_km
        x_len = side_len_km
    if start_corner[1] == 1:
        y_spacing = -resolution_km
        y_len = -side_len_km
    else:
        y_spacing = resolution_km
        y_len = side_len_km
    
    east_grid = np.arange(np.round(corner_utm[0], round_km), corner_utm[0] + x_len, x_spacing)
    north_grid = np.arange(np.round(corner_utm[1], round_km), corner_utm[1] + y_len, y_spacing)
    return east_grid, north_grid


def init_ds(time, easting, northing):
    # Create the initial dataset
    ds_int = xr.Dataset(coords={'time': time, 'easting': easting, 'northing': northing})
    ds_int['easting'].attrs['units'] = 'km'
    ds_int['northing'].attrs['units'] = 'km'
    ds_int.attrs['zone'] = 51
    ds_int.attrs['proj'] = 'UTM'
    ds_int.attrs['ellps'] = 'WGS84'
    ds_int.attrs['south'] = 1
    return ds_int