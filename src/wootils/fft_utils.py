# import pyproj as pp
import numpy as np
# import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




def calc_fundamental(dimvals, safety=1, dt64_units='s', ):
    '''
    Calc fundamental freq approx. of a 1D series
    in cycles per specified unit (defaults to seconds for time)
    '''
    # Check if dimvals is time or float
    if np.isscalar(dimvals) and isinstance(dimvals, (float, int, np.floating, np.integer)):
        # Calc fundamental freq
        funfq = safety/dimvals
    elif np.issubdtype(dimvals.dtype, np.datetime64):
        # Calc fundamental freq
        funfq = safety/(1 * len(dimvals) * (np.diff(dimvals)[0] / np.timedelta64(1,dt64_units))
                        * (np.timedelta64(1,dt64_units).astype('int')))
    elif np.issubdtype(dimvals.dtype, np.float64) | np.issubdtype(dimvals.dtype, np.int64):
        # Calc fundamental freq
        funfq = safety/(dimvals[-1] - dimvals[0])
    else:
        raise(Exception("\'dimvals\' must be time or float"))
    return funfq


def calc_nyquist(dimvals, safety=2, dt64_units='s'):
    '''
    Calc nyquist approx. in cycles per unit of dimension (defaults to seconds for time)
    dimvals: dimension array (time or float)
    safety: factor to divide by (min. 2, higher for noisey data)
    '''
    # Check if dimvals needs to be squeezed
    if len(dimvals.shape) > 1:
        dimvals = np.squeeze(dimvals)
    # Check if dimvals is time or float
    if np.isscalar(dimvals) and isinstance(dimvals, (float, int, np.floating, np.integer)):
        return 1/(safety * dimvals)
    elif np.issubdtype(dimvals.dtype, np.datetime64):
        return 1/(safety * (np.diff(dimvals)[0] / np.timedelta64(1,dt64_units)))
    elif np.issubdtype(dimvals.dtype, np.float64) | np.issubdtype(dimvals.dtype, np.int64):
        return 1/(safety * np.diff(dimvals)[0])












def fit_quadratic_transect(x_gridded, z_points):
    x_gridded = x_gridded.flatten()
    A = np.array([x_gridded*0+1, x_gridded, x_gridded**2]).T
    B = z_points.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
    return coeff

def reconstruct_transect(x_gridded, coeff):
    int_arr = (coeff[0] + x_gridded*coeff[1] + x_gridded**2*coeff[2])
    return int_arr.reshape(x_gridded.shape)

def remove_barotropic_xr_transect(xr_array, xdim='easting', tdim='time', progress=True):
    '''Loop over time dimension and remove the barotropic component by quadratic surface fit'''
    x = xr_array[xdim].values
    xg, yg = np.meshgrid(x, [0])

    # Re-order so time is first
    xr_array = xr_array.transpose(tdim, xdim)
    z = xr_array.values
    z_int = np.zeros_like(z)

    for i in tqdm(range(len(xr_array[tdim].values)), disable=not progress):
        coeff = fit_quadratic_transect(xg, z[i,:])
        z_int[i,:] = z[i,:] - reconstruct_transect(xg, coeff)

    xr_array_int = xr_array.copy(data=z_int)
    return xr_array_int


# def fit_quadratic_surface(x_gridded, y_gridded, z_points):
#     x_gridded = x_gridded.flatten()
#     y_gridded = y_gridded.flatten()

#     A = np.array([x_gridded*0+1, x_gridded, y_gridded, x_gridded**2, x_gridded**2*y_gridded,
#                   x_gridded**2*y_gridded**2, y_gridded**2, x_gridded*y_gridded**2, x_gridded*y_gridded]).T
#     B = z_points.flatten()

#     coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
#     return coeff

# def reconstruct_surface(x_gridded, y_gridded, coeff):
#     int_arr = (coeff[0] + x_gridded*coeff[1] + y_gridded*coeff[2] + x_gridded**2*coeff[3] +
#                x_gridded**2*y_gridded*coeff[4] + x_gridded**2*y_gridded**2*coeff[5] + 
#                y_gridded**2*coeff[6] + x_gridded*y_gridded**2*coeff[7] + x_gridded*y_gridded*coeff[8])
#     return int_arr.reshape(x_gridded.shape)

# def remove_barotropic_xr(xr_array, xdim='easting', ydim='northing', tdim='time', progress=True):
#     '''Loop over time dimension and remove the barotropic component by quadratic surface fit'''
#     x = xr_array[xdim].values
#     y = xr_array[ydim].values
#     xg, yg = np.meshgrid(x, y)

#     # Re-order so time is first
#     xr_array = xr_array.transpose(tdim, ydim, xdim)
#     z = xr_array.values
#     z_int = np.zeros_like(z)

#     for i in tqdm(range(len(xr_array[tdim].values)), disable=not progress):
#         coeff = fit_quadratic_surface(xg, yg, z[i,:,:])
#         z_int[i,:,:] = z[i,:,:] - reconstruct_surface(xg, yg, coeff)

#     xr_array_int = xr_array.copy(data=z_int)
#     return xr_array_int


def animate_da(da, season, var, vmin=-0.25, vmax=0.25, cmap='PuOr', hsize=7, vsize=5, app=None):
    fig, ax = plt.subplots(1, 1, figsize=(hsize, vsize))
    da.isel(time=0).plot(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, cbar_kwargs={'pad':0.01}) 
    ax.set_title(f'SUNTANS surface {var} in {season}: {str(da.time.values[0])[:13]}')
    ax.set_ylabel('Northing [km]')
    ax.set_xlabel('Easting [km]')
    def update(tstep):
        ax.clear()
        da.isel(time=tstep).plot(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=False)
        ax.set_title(f'SUNTANS surface {var} in {season}: {str(da.time.values[tstep])[:13]}')
    anim = FuncAnimation(fig, update, frames=range(len(da['time'])))
    anim.save(f'SUNTANS_{season}_{var}_{app}.mp4', writer='ffmpeg', fps=24)
    return anim


# def fit_quadratic_surface(x, y, z):
#     """
#     Fit a quadratic surface to data (structured or unstructured).
    
#     Parameters:
#     x, y : array-like
#         Coordinates of the data points.
#     z : array-like
#         Values at the data points.
    
#     Returns:
#     coeff : ndarray
#         Coefficients of the fitted quadratic surface.
#     """
#     x = np.asarray(x).flatten()
#     y = np.asarray(y).flatten()
#     # z = np.asarray(z).flatten()

#     A = np.array([x*0+1, x, y, x**2, x**2*y, x**2*y**2, y**2, x*y**2, x*y]).T
#     B = z

#     coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
#     return coeff


# def reconstruct_surface(x, y, coeff):
#     """
#     Reconstruct the quadratic surface from the coefficients.
    
#     Parameters:
#     x, y : array-like
#         Coordinates of the data points.
#     coeff : ndarray
#         Coefficients of the fitted quadratic surface.
    
#     Returns:
#     z_reconstructed : ndarray
#         Reconstructed values at the data points.
#     """
#     z_reconstructed = (coeff[0] + x*coeff[1] + y*coeff[2] + x**2*coeff[3] +
#                        x**2*y*coeff[4] + x**2*y**2*coeff[5] + y**2*coeff[6] +
#                        x*y**2*coeff[7] + x*y*coeff[8])
#     return z_reconstructed


# def remove_barotropic_xr(xr_array, xdim='easting', ydim='northing', tdim='time', ds=None, progress=True):
#     """
#     Loop over time dimension and remove the barotropic component by quadratic surface fit.
    
#     Parameters:
#     xr_array : xarray.DataArray
#         Input data array.
#     xdim : str
#         Name of the x dimension.
#     ydim : str
#         Name of the y dimension.
#     tdim : str
#         Name of the time dimension.
#     progress : bool
#         Whether to show a progress bar.
    
#     Returns:
#     xr_array_int : xarray.DataArray
#         Data array with the barotropic component removed.
#     """
#     if ds is None:
#         x = xr_array[xdim].values
#         y = xr_array[ydim].values
#     else:
#         x = ds[xdim].values
#         y = ds[ydim].values
#     xg, yg = np.meshgrid(x, y)

#     # Re-order so time is first
#     # xr_array = xr_array.transpose(tdim, ydim, xdim)
#     z = xr_array.values
#     z_int = np.zeros_like(z)

#     for i in tqdm(range(len(xr_array[tdim].values)), disable=not progress):
#         coeff = fit_quadratic_surface(xg, yg, z[i,:].T)
#         z_int[i,:] = z[i,:].T - reconstruct_surface(xg, yg, coeff)

#     xr_array_int = xr_array.copy(data=z_int)
#     return xr_array_int