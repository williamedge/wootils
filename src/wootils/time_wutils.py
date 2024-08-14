import pandas as pd
import numpy as np
import xarray as xr


def get_limidxs(time, start, end):
    # Check if xarray
    if isinstance(time, xr.DataArray):
        time = time.values
    lim_idxs = (time>=start) & (time<=end)
    return lim_idxs

def xr_wellspaced(xr_obj, var='time'):
    common_space = np.median(np.diff(xr_obj[var].values))
    return np.all(np.diff(xr_obj[var].values) == common_space)

def time_median_missing(time_arr):
    median_step = np.median(np.diff(time_arr))
    missing_median = np.sum(np.diff(time_arr) != median_step)
    return median_step, missing_median



def month_index(time):
    return pd.DatetimeIndex(pd.Series(time)).month


def season_index(month_idx, hemisphere='S'):

    if hemisphere=='N':
        season_dict = {'1': 'Winter',
                    '2': 'Winter',
                    '3': 'Spring', 
                    '4': 'Spring',
                    '5': 'Spring',
                    '6': 'Summer',
                    '7': 'Summer',
                    '8': 'Summer',
                    '9': 'Autumn',
                    '10': 'Autumn',
                    '11': 'Autumn',
                    '12': 'Winter'}
    else:
        season_dict = {'1': 'Summer',
                    '2': 'Summer',
                    '3': 'Autumn', 
                    '4': 'Autumn',
                    '5': 'Autumn',
                    '6': 'Winter',
                    '7': 'Winter',
                    '8': 'Winter',
                    '9': 'Spring',
                    '10': 'Spring',
                    '11': 'Spring',
                    '12': 'Summer'}

    return month_idx.to_series().apply(lambda x: season_dict[str(x)]).values

