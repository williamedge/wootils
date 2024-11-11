import numpy as np
from scipy.stats import norm


def halfnormal_std(x):
    return np.sqrt(np.sum(2*x**2) / (2*len(x)))


def halfnormal_nanstd(x):
    return np.sqrt(np.nansum(2*x**2) / (2*len(x)))


def psd_magnitude(psd, freq, psd_freq, coeff=False, scaling='density'):
    assert scaling in ['density', 'spectrum']
    f_ix = np.argmin(np.abs(psd_freq[:, np.newaxis] - freq), axis=0)
    if coeff:
        mult = 1
    else:
        mult = 2 * np.pi

    if scaling == 'density':
        scaling = psd_freq[f_ix]
    else:
        scaling = 1
    if len(psd.shape) == 1:
        return np.sqrt(2 * scaling * psd[f_ix]) * mult
    else:
        return np.sqrt(2 * np.expand_dims(scaling, axis=(1,0)).T * psd[f_ix]) * mult


def psd_mag_xr(psd, freq, freq_dim='time', coeff=False, scaling='density'):
    return psd_magnitude(psd.values, freq, psd['freq_' + freq_dim].values, coeff=coeff, scaling=scaling)



def zero_crossing(x, acf):
    x_ix = x >= 0 
    return x[x_ix][np.where(np.diff(np.sign(acf[x_ix])))[0][0]]


def acf_confidence(N, alpha=0.05):
    T = np.arange(1,N//2)
    z_alpha = norm.ppf(1 - alpha / 2)
    confint = z_alpha / np.sqrt(N - T)
    return confint**2


def sig_crossing(x, acf, alpha=0.05):
    N_acf = len(acf)
    acf_sif = acf_confidence(N_acf, alpha)
    ix = np.where(acf.real[N_acf//2:(N_acf//2 + len(acf_sif)+1)][1:].values < acf_sif)[0][0] + 1
    return x[N_acf//2:][ix]
