import numpy as np


def calc_nyquist(timestep, safety=2):
    '''
    Calc nyquist approx. in CPD (using 1/3 del.t)
    timestep: timestep in minutes
    safety: factor to divide by (min. 2, higher for noisey data)
    '''
    return 1/(safety * (np.timedelta64(timestep,'m').astype('int')/(60*24)))


def calc_inertial(latitude):
    '''
    Get inertial frequency (Ω = 7.2921 × 10−5 rad/s)
    latitude: in decimal degrees (sign unimportant?)
    '''
    f = 2 * 7.2921e-5 * np.sin(np.deg2rad(latitude))
    cpd = (24*60*60) / (2*np.pi / -f)
    return f, cpd