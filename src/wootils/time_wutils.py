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


def nearest_hour(time_xr, hour_step_int):
    hr = time_xr[time_xr.dt.hour % hour_step_int == 0][0]
    return hr.astype('datetime64[D]') + np.timedelta64(int(hr.dt.hour), 'h')


def nan_wraptime(da, nan_days=[0,364,365,366]):
    for n in nan_days:
        da.loc[da.time.dt.dayofyear == n] = np.nan
    return da


def month_index(time):
    return pd.DatetimeIndex(pd.Series(time)).month


def yawuru_names():
    yawuru_seasons = np.array(['Man-gala', 'Marrul', 'Wirralburu', 'Barrgana', 'Wirlburu', 'Laja'])
    yawuru_sstr = ['Dec-Mar', 'Apr', 'May', 'Jun-Aug', 'Sep', 'Oct-Nov']
    return yawuru_seasons, yawuru_sstr


def season_index(month_idx, seasons='S'):

    if seasons=='N':
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
    elif seasons=='S':
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
    elif seasons == 'yawuru':
        season_dict = {'1': 'Man-gala',
                    '2': 'Man-gala',
                    '3': 'Man-gala', 
                    '4': 'Marrul',
                    '5': 'Wirralburu',
                    '6': 'Barrgana',
                    '7': 'Barrgana',
                    '8': 'Barrgana',
                    '9': 'Wirlburu',
                    '10': 'Laja',
                    '11': 'Laja',
                    '12': 'Man-gala'}

    return month_idx.to_series().apply(lambda x: season_dict[str(x)]).values


def is_regular_time_xr(da, step=None):
    """Check if the time dimension is regular. Check if it matches a given step."""
    time_diff = np.diff(da['time'].values)
    regular = np.all(time_diff == time_diff[0])
    if step is not None:
        regular &= time_diff[0] == step
    return regular
