import numpy as np
import xarray as xr
from scipy.signal import sosfiltfilt, butter



def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def check_spacing(time):
    tdiff = np.diff(time)
    return all_equal(tdiff)


def edge_nanx(data, reverse):
    lead_nans = False
    nanx = np.where(~np.isnan(data))[0]
    if len(nanx) > 0:
        if nanx[0] != 0:
            lead_nans = True
            nan_ix = nanx[0]
        else:
            nan_ix = nanx[0]
    else:
        nan_ix=0
    if reverse:
        nan_ix = len(data) - nan_ix
    return lead_nans, nan_ix


def trim_edge_nans_xr(data_array):
    # Time should be first dimension for 2D arrays
    lead_nan, nan_lx = edge_nanx(data_array, False)
    end_nan, nan_ex = edge_nanx(data_array[::-1], True)
    return data_array[nan_lx:nan_ex], nan_lx, nan_ex


def max_gap_xr(data, timedim):
    nanx = np.isnan(data)
    time = data[timedim][~nanx]
    gaps = np.diff(time).astype('timedelta64[m]')
    return np.max(gaps), np.argmax(gaps)


def get_break_idxs(xr_data, timevar, limit_mins):
    wh_nan = np.where(~np.isnan(xr_data))[0]
    nanx = ~np.isnan(xr_data)
    t_diff = np.diff(xr_data[timevar][nanx]).astype('timedelta64[m]')
    ix_lim = np.where(t_diff.astype('int') > limit_mins)[0]
    ix_lim = ix_lim + 1
    ix_lim = np.append(0, ix_lim)
    ix_lim = np.append(ix_lim, len(xr_data[nanx])-1)
    return wh_nan[ix_lim]

def filter1d(data, filt_hours, tstep_minutes, ftype='lowpass', axis=-1):
    sos = butter(2, 1/(filt_hours*60*60), btype=ftype, output='sos', fs=1/(60*tstep_minutes))
    data_out = sosfiltfilt(sos, data, axis=axis)
    return data_out  

def filter1d_xr(data, filt_hours, tstep_minutes, ftype='lowpass', axis=-1):
    data_out = data.copy()
    sos = butter(2, 1/(filt_hours*60*60), btype=ftype, output='sos', fs=1/(60*tstep_minutes))
    data_out[:] = sosfiltfilt(sos, data, axis=axis)
    return data_out


def even_butter(time, data, tstep_sec=None, filt_seconds=1, order=4, ftype='lowpass'):
    if tstep_sec is None:
        tstep_sec = np.diff(time)[0] / np.timedelta64(1,'s')
    sos = butter(order, 1/(filt_seconds), btype=ftype, output='sos', fs=1/(tstep_sec))
    data_out = sosfiltfilt(sos, data, axis=0)
    return data_out  
