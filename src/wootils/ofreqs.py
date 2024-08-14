import numpy as np


def calc_nyquist(timestep, safety=2):
    '''
    Calc nyquist approx. in CPD (using 1/2 del.t)
    timestep: timestep in minutes
    safety: factor to divide by (min. 2, higher for noisey data)
    '''
    return 1/(safety * (np.timedelta64(timestep,'m').astype('int')/(60*24)))


def calc_fundamental(dimvals, data, dt64_units='s', safety=1):
    '''
    Calc fundamental freq approx. of a 1D series
    in cycles per specified unit (defaults to seconds)
    '''
    # Calc true length of data by removing nans (is this ok?)
    # raise(Exception("This function is not yet implemented"))
    bad_len = np.sum(~np.isnan(data), axis=0)
    true_len = np.diff(dimvals.values)[0] * bad_len

    # Check if dimvals is time or float
    if np.issubdtype(dimvals.dtype, np.datetime64):
        # Calc fundamental freq
        funfq = safety/(1 * true_len * (np.timedelta64(1,dt64_units).astype('int')))
    elif np.issubdtype(dimvals.dtype, np.float64):
        # Calc fundamental freq
        funfq = safety/(true_len)
    else:
        raise(Exception("\'dimvals\' must be time or float"))

    # # Calc fundamental freq approx. in CPD (using 3/2 n del.t?)
    # funfq = safety/(1 * true_len.values * (np.timedelta64(1,'m').astype('int')/(60*24)))
    return funfq


def calc_inertial(latitude):
    '''
    Get inertial frequency (Ω = 7.2921 × 10−5 rad/s)
    latitude: in decimal degrees (sign unimportant?)
    '''
    f = 2 * 7.2921e-5 * np.sin(np.deg2rad(latitude))
    cpd = (24*60*60) / (2*np.pi / -f)
    return f, cpd