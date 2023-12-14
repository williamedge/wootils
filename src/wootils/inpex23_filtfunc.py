import xarray as xr
import numpy as np 
import wootils.filters as fl
from afloat.timeseries import quick_harmonic
from wootils.fitters import murphy_ss



def extract_and_flag(data, flag):
    data_new = data.copy()
    data_new[flag] = np.nan
    return data_new


def create_time(nc_files):
    ds1 = xr.open_dataset(nc_files[0])
    dse = xr.open_dataset(nc_files[-1])

    time_all = np.arange(ds1['time'][0].values, dse['time'][-1].values + np.timedelta64(1,'m'),\
                        np.timedelta64(1,'m'))
    z = ds1['z_nom'].values
    # time_len = (time_all[-1] - time_all[0]).astype('timedelta64[m]')
    return time_all, z


def load_currents(nc_files):

    time_all, z = create_time(nc_files)

    data_u_all = np.full((len(time_all), len(z)), np.nan)
    data_v_all = np.full((len(time_all), len(z)), np.nan)

    for dgr in nc_files:
        ds = xr.open_dataset(dgr)

        tx = (time_all >= ds['time'][0].values) & (time_all <= ds['time'][-1].values)
        
        data_u_all[tx,:] = ds['water_u'].T
        data_v_all[tx,:] = ds['water_v'].T

    data_u = xr.DataArray(data=data_u_all, coords={'time':time_all, 'z':z})
    data_v = xr.DataArray(data=data_v_all, coords={'time':time_all, 'z':z})
    ds = data_u.to_dataset(name='u_meas')
    ds['v_meas'] = data_v
    return ds


def load_temperature(nc_files):

    time_all, z = create_time(nc_files)

    data_u_all = np.full((len(time_all), len(z)), np.nan)

    for dgr in nc_files:
        ds = xr.open_dataset(dgr)

        tx = (time_all >= ds['time'][0].values) & (time_all <= ds['time'][-1].values)
        
        data_u_all[tx,:] = ds['temperature'].T

    data_u = xr.DataArray(data=data_u_all, coords={'time':time_all, 'z':z})
    ds = data_u.to_dataset(name='temperature')
    return ds


def load_amp(nc_files, mode=0):

    time_all, z = create_time(nc_files)

    data_u_all = np.full((len(time_all),), np.nan)

    for dgr in nc_files:
        ds = xr.open_dataset(dgr)
        tx = (time_all >= ds['time'][0].values) & (time_all <= ds['time'][-1].values)
        data_u_all[tx] = ds['A_n'].sel(modes=mode)

    data_u = xr.DataArray(data=data_u_all, coords={'time':time_all})
    return data_u


def zero_pad(x_arr, int_minutes=60):
    
    # Interpolate small gaps and zero-pad the rest
    data_u_na = x_arr.interpolate_na(dim='time', max_gap=np.timedelta64(int_minutes,'m'))

    # # Save the nans
    # u_nan = np.isnan(data_u_na)
    # v_nan = np.isnan(data_v_na)

    # requires conversion back to numpy for 2D indexing
    data_u_np = data_u_na.values
    data_u_np[np.isnan(data_u_np)] = 0.0

    # back to xarray
    if 'z' in data_u_na.coords:
        data_u_zp = xr.DataArray(data=data_u_np, coords={'time':x_arr.time, 'z':x_arr.z})
    elif 'z_nom' in data_u_na.coords:
        data_u_zp = xr.DataArray(data=data_u_np, coords={'time':x_arr.time, 'z_nom':x_arr.z_nom})
    else:
        data_u_zp = xr.DataArray(data=data_u_np, coords={'time':x_arr.time})

    return data_u_zp


def nans_back(new, old):
    new_np = new.values.copy()
    new_np[np.isnan(old)] = np.nan
    return xr.DataArray(data=new_np, coords=new.coords)

def zeros_back(data, nans):
    dcoords = data.coords
    data = data.values.copy()
    data[np.isnan(nans)] = 0.0
    return xr.DataArray(data=data, coords=dcoords)


def get_tidal_currents(U, V, filt_low=30, filt_high=6):

    if filt_low is not None:
        u_tide = xr.apply_ufunc(fl.filter1d, U, filt_low, 1, 'highpass',\
                        input_core_dims=[['time'],[],[],[]], output_core_dims=[['time']],\
                        vectorize=True)
        v_tide = xr.apply_ufunc(fl.filter1d, V, filt_low, 1, 'highpass',\
                        input_core_dims=[['time'],[],[],[]], output_core_dims=[['time']],\
                        vectorize=True)
    else:
        u_tide = U
        v_tide = V

    if filt_high is not None:
        u_tide = xr.apply_ufunc(fl.filter1d, u_tide, filt_high, 1, 'lowpass',\
                        input_core_dims=[['time'],[],[],[]], output_core_dims=[['time']],\
                        vectorize=True)
        v_tide = xr.apply_ufunc(fl.filter1d, v_tide, filt_high, 1, 'lowpass',\
                        input_core_dims=[['time'],[],[],[]], output_core_dims=[['time']],\
                        vectorize=True)

    ds = u_tide.to_dataset(name='u_tide')
    ds['v_tide'] = v_tide
    return ds


def get_tidal_amp(amp, filt_low=30, filt_high=6):

    if filt_low is not None:
        u_tide = xr.apply_ufunc(fl.filter1d, amp, filt_low, 1, 'highpass',\
                        input_core_dims=[['time'],[],[],[]], output_core_dims=[['time']],\
                        vectorize=True)

    else:
        u_tide = amp

    if filt_high is not None:
        u_tide = xr.apply_ufunc(fl.filter1d, u_tide, filt_high, 1, 'lowpass',\
                        input_core_dims=[['time'],[],[],[]], output_core_dims=[['time']],\
                        vectorize=True)

    return u_tide



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