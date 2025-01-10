import numpy as np
import xarray as xr
from wootils.wootxr import xrwrap
from wootils.uwindows import *


@xr.register_dataarray_accessor("ufilt")
class ufilters(xrwrap):
    '''
    xarray-based class for filtering unevenly spaced data in 1 or more dimensions.

    called function is always centred. 
    '''

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def urolling(self, roll_dim, window_size, min_periods=None, coord_thin=1, new_coord=None):
        if isinstance(self._obj, xr.Dataset):
            # return DatasetRolling(self._obj, roll_dim, window_size, coord_thin, new_coord)
            raise NotImplementedError("DatasetRolling not implemented.")
        elif isinstance(self._obj, xr.DataArray):
            return DataArrayRolling(self._obj, roll_dim, window_size, min_periods=min_periods,
                                    coord_thin=coord_thin, new_coord=new_coord)
        else:
            raise TypeError("Unsupported type. Expected xarray.Dataset or xarray.DataArray.")


class BaseRolling:
    def __init__(self, obj, roll_dim, window_size, **kwargs):
        self._obj = obj
        self.roll_dim = roll_dim
        self.window_size = window_size
        self.min_periods = kwargs.get('min_periods', None)
        assert self.min_periods != 0 # min_periods cannot be 0
        # self.min_periods = kwargs.get('min_periods', None)
        self.coord_thin = kwargs.get('coord_thin', 1)
        self.new_coord = kwargs.get('new_coord', None)


    def get_min_periods(self, full_window):      
        # Set the minimum number of periods in the window
        if self.min_periods is None:
            self.min_periods = self.max_periods_in_window(self._obj[self.roll_dim].values, full_window)
        # If min_periods is between 0 and 1, calculate it as a percentage of max_periods
        elif 0 < self.min_periods < 1:
            self.min_periods = int(self.min_periods *
                                   self.max_periods_in_window(self._obj[self.roll_dim].values, full_window))


    def boxcar(self, **kwargs):
        self.type = 'boxcar'

        # Set the minimum number of periods in the window
        self.get_min_periods(self.window_size)
            
        # Calculate the rolling window indexes
        self.rolling_indexes = self.boxcar_index(self._obj[self.roll_dim].values,
                                                 self.window_size,
                                                 self.coord_thin,
                                                 self.new_coord)
        return self
    
    
    def savgol(self, weights='boxcar', **kwargs):
        if weights == 'boxcar':
            self = self.boxcar()
        elif weights == 'gaussian':
            self = self.gaussian()
        self.type = 'savgol'
        self.wtype = weights
        self.polyorder = kwargs.get('polyorder', 2)
        return self        
        

    def gaussian(self, truncate=4, **kwargs):
        self.type = 'gaussian'
        self.truncate = truncate

        # Set the minimum number of periods in the window
        self.get_min_periods(2 * self.window_size * self.truncate)
            
        # Calculate the rolling window indexes
        self.rolling_indexes = self.gaussian_index(self._obj[self.roll_dim].values,
                                                   self.window_size,
                                                   self.truncate,
                                                   self.coord_thin,
                                                   self.new_coord)
        return self
    
    def apply(self, func, **kwargs):
        data = self._obj.values
        rolled_data = np.full((len(self.rolling_indexes),) + data.shape[1:], np.nan)

        # Use new_coord if supplied, otherwise use the thinned original coord
        if self.new_coord is not None:
            coords = {self.roll_dim: self.new_coord}
        else:
            coords = {self.roll_dim: self._obj[self.roll_dim].values[::self.coord_thin]}

        for i, (start, end) in enumerate(self.rolling_indexes):
            if np.count_nonzero(~np.isnan(data[start:end])) >= self.min_periods:
                # Remove the nan values
                nanx = ~np.isnan(data[start:end])
                # Calculate the function weights
                if (self.type == 'boxcar') or (self.wtype == 'boxcar'):
                    weights = np.ones_like(data[start:end][nanx])
                elif (self.type == 'gaussian') or (self.wtype == 'gaussian'):
                    weights = gaussian_uweights(coords[self.roll_dim][i],
                                                self._obj[self.roll_dim][start:end][nanx].values,
                                                self.window_size)
                if self.type == 'savgol':
                    # Apply the Savitzky-Golay filter to the data within the window
                    time = self._obj[self.roll_dim][start:end][nanx].values - coords[self.roll_dim][i]
                    rolled_data[i] = self.savgol_ufilt(time,
                                                       data[start:end][nanx],
                                                       weights,
                                                       self.polyorder)[0]
                else:
                    # Apply the rolling function to the data within the window
                    rolled_data[i] = func(data[start:end][nanx] * weights, **kwargs)

        # Preserve other dimensions and coordinates
        for dim in self._obj.dims:
            if dim != self.roll_dim:
                coords[dim] = self._obj[dim]

        return xr.DataArray(rolled_data, dims=self._obj.dims, coords=coords, attrs=self._obj.attrs)


    @staticmethod
    def boxcar_index(coord, windowsize, coordthin=1, newcoord=None):
        """
        Determines the indices for sliding windows (borrowed from afloat). 

        The rolling dim does not need to be evenly spaced.
        
        Parameters
        ----------
        coord: list or np.array 
            Coordinate vector input as list or array containing coordinate locs. 

        windowsize: numeric 
            Length of the window [can be any dtype that supports +-]

        coordthin: int
            Number of points to thin the coordinate vector by.

        newcoord: list or np.array
            A new coord vector to use for the window index. If None, the function
            will thin the input coord vector. Cannot be used with coord_thin.
            
        Returns
        -------
        window_indexes : list
            A 2D list containing the start and end indices for each window.
            Will have a length equal to new_coord if used, otherwise the thinned original
            coord.        
        """

        if newcoord is not None:
            assert coordthin == 1, "Cannot use coord_thin with new_coord"

        # Set the coord to iterate over
        if newcoord is None:
            loop_coord = np.copy(coord)[::coordthin]
        else:
            loop_coord = newcoord

        # Initialize the window indexes
        window_indexes = []

        # Iterate over the coord vector
        for i in range(len(loop_coord)):
            # Calculate the start and end indices for the window
            start = np.searchsorted(coord, loop_coord[i] - windowsize / 2)
            end = np.searchsorted(coord, loop_coord[i] + windowsize / 2)

            # Append the window indexes
            window_indexes.append([start, end])

        return window_indexes


    @staticmethod
    def gaussian_index(coord, sigma, truncate=4, coordthin=1, newcoord=None):
        """
        Determines the indices for sliding windows using a Gaussian kernel.

        The rolling dim does not need to be evenly spaced.

        Parameters
        ----------
        coord: list or np.array
            Coordinate vector input as list or array containing coordinate locs.

        sigma: float
            Standard deviation of the Gaussian kernel.

        truncate: float, optional
            Truncate the Gaussian kernel beyond this many standard deviations.
            Default is 4.0.
            
        coordthin: int
            Number of points to thin the coordinate vector by.

        newcoord: list or np.array
            A new coord vector to use for the window index. If None, the function
            will thin the input coord vector. Cannot be used with coord_thin.

        Returns
        -------
        window_indexes : list
            A 2D list containing the start and end indices for each window.
            Will have a length equal to new_coord if used, otherwise the thinned original
            coord.
        """

        if newcoord is not None:
            assert coordthin == 1, "Cannot use coord_thin with new_coord"

        # Set the coord to iterate over
        if newcoord is None:
            loop_coord = np.copy(coord)[::coordthin]
        else:
            loop_coord = newcoord

        # Initialize the window indexes
        window_indexes = []

        # Iterate over the coord vector
        for i in range(len(loop_coord)):
            # Calculate the start and end indices for the window based on the truncate arg
            center = loop_coord[i]
            start = np.searchsorted(coord, center - truncate * sigma)
            end = np.searchsorted(coord, center + truncate * sigma)
            
            # Append the window indexes
            window_indexes.append([start, end])

        return window_indexes


    @staticmethod
    def max_periods_in_window(coord, window_size):
        """
        Calculate the maximum periods expected in a window based on the median step of the coordinate.

        Parameters
        ----------
        coord : list or np.array
            Coordinate vector input as list or array containing coordinate locs.
        window_size : numeric
            Length of the window [can be any dtype that supports +-]

        Returns
        -------
        max_periods : int
            Maximum number of periods expected in a window.
        """
        # Calculate the differences between consecutive elements
        steps = np.diff(coord)
        
        # Compute the median of these differences
        median_step = np.median(steps)
        
        # Calculate the maximum periods
        max_periods = int(window_size / median_step)
        
        return max_periods
    
    @staticmethod
    def savgol_ufilt(time, data, weights, polyorder=2):
        '''
        Apply an unstructured Savitzky-Golay filter to the data. 
        '''
        # Set up the Vandermonde matrix for the polynomial fit
        A = np.vander(time, N=polyorder + 1, increasing=True)

        # Solve the weighted least-squares problem
        W = np.diag(weights)
        ATA = A.T @ W @ A
        ATy = A.T @ W @ data
        c = np.linalg.solve(ATA, ATy)  # Polynomial coefficients
        return c
    


class DataArrayRolling(BaseRolling):
    def __init__(self, xarray_obj, roll_dim, window_size, **kwargs):
        super().__init__(xarray_obj, roll_dim, window_size, **kwargs)
