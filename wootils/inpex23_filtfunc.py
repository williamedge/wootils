import xarray as xr
import numpy as np 
import wutils.filters as fl
from afloat.timeseries import quick_harmonic
from wutils.fitters import murphy_ss

def extract_and_flag(data, flag):
    data_new = data.copy()
    data_new[flag] = np.nan
    return data_new


def trim_interp_filter_xa(data_array, timedim, filt_hours, filttype, intlim_mins):
    # Function to trim edge nans, interpolate nans within data, and then filter
    # returns an xarray object with the same length as the input, with edge nans kept

    # data_all = data_array.copy()
    # data_all[:] = np.nan
    data_all = xr.DataArray(data=np.full(data_array.values.shape, np.nan),\
                                    coords={timedim:data_array[timedim]})

    # Remove leading or trailing nans
    data_trim, nan_lx, nan_ex = fl.trim_edge_nans_xr(data_array)

    # Check remaining nans don't exceed interpolation limit
    mgap, igap = fl.max_gap_xr(data_trim, timedim)
    if mgap.astype('timedelta64[m]').astype(int) > intlim_mins:
        raise ValueError('Max gap in time series data is longer than allowed by user.')

    else:
        # Check array is well-spaced
        ch_spce = fl.check_spacing(data_trim[timedim].values)

        if not ch_spce:
            raise ValueError('Array is not evenly spaced.')
        else:
            # Get time step in minutes
            tstep = np.diff(data_trim[timedim].values)[0].astype('timedelta64[m]').astype(int)
            
            # Interpolate nans
            data_trim_na = data_trim.interpolate_na(timedim)

            # Filter data
            data_trim_lp = fl.filter1d_xr(data_trim_na, filt_hours, tstep, ftype=filttype)
    
    data_all[nan_lx:nan_ex] = data_trim_lp
    return data_all



def filter_dropin_xa(data_array, timedim, filt_hours, filttype, intlim_mins):
    
    # data_all = data_array.copy()
    # data_all[:] = np.nan

    # Remove leading or trailing nans
    data_trim, nan_lx, nan_ex = fl.trim_edge_nans_xr(data_array)

    data_all = xr.DataArray(data=np.full(data_trim.values.shape, np.nan),\
                                  coords={timedim:data_trim[timedim]})
    
    # Check remaining nans don't exceed interpolation limit
    mgap, igap = fl.max_gap_xr(data_trim, 'time')

    if mgap.astype('timedelta64[m]').astype(int) > intlim_mins:
        print('Max gap in time series data is longer than allowed by user. Splitting dataset')

        # Get indices of break points
        splt_ix = fl.get_break_idxs(data_trim, timedim, intlim_mins)

        # Initiate array to fill back in
        comb_data = xr.DataArray(data=np.full(data_trim.values.shape, np.nan),\
                                  coords={timedim:data_trim[timedim]})

        # loop through partial data series
        for ii, ff in zip(splt_ix[:-1], splt_ix[1:]):

            # Trim edge nans, interp middle nans, filter, and put back into array
            comb_data[ii:ff] = trim_interp_filter_xa(data_trim[ii:ff], timedim, filt_hours, filttype, intlim_mins)

    else:
        # Lowpass filter data
        comb_data = trim_interp_filter_xa(data_trim, timedim, filt_hours, filttype, intlim_mins)

    # Drop back into full time series
    data_all = comb_data.values
    
    return data_all


def probabilistic_harmonic_fit(time, data, flag, window, no_iters, freqs, data_tolerance):
    '''
    Function to perform a simple harmonic fit multiple times in a while loop.
    Randomly selects a location within the timeseries, checks the data within the window
    is good (within tolerance), and fits a harmonic with the chosen frequencies and
    outputs the skill score of the fit.

    Inputs
    ------
    time:       datetime64 timeseries (xarray or numpy) (assumes well-spaced for now)
    data:       1D data (xarray or numpy)
    flag:       boolean array flagged True for bad data
    window:     number of days to fit to (int)
    no_iters:   number of iterations to collect

    Outputs
    -------
    skill_score: numpy array of calculated skill scores from harmonic fit (length=no_iters)
    '''

    # Try and convert time to numpy
    try:
        time = time.values
    except:
        time = time

    try:
        data = data.values
    except:
        data = data

    try:
        flag = flag.values
    except:
        flag = flag    

    fitlen = np.timedelta64(window,'D')

    # Get data timestep and set the min. length of data in a window
    timestep = np.diff(time)[0]
    goodlen = (data_tolerance * window * (np.timedelta64(1,'D')/timestep)).astype('int')

    # Initiate output array
    skill_score = np.full(no_iters, np.nan)

    # Start while loop with count
    count = 0
    while count < no_iters:

        # Select a random point in time series
        rint = np.random.randint(0, len(time))

        # Get the data within time window
        txx = (time >= time[rint]) &\
              (time <= time[rint] + fitlen)
        
        # Check length
        if len(time[txx]) > goodlen:

            # Check data quality
            tmpflg = ~flag[txx]
            if np.sum(tmpflg) > goodlen:

                # Proceed with fit
                qh_tmp = quick_harmonic(time[txx], data[txx], freqs)

                # calculate skill score
                skill_score[count] = murphy_ss(data[txx], qh_tmp['data_fit'])
                count += 1

    return skill_score