import numpy as np
from scipy import signal


def coherence(f, sig_x, sig_y, Fs, nfft=None):
    fqall, csd_x = signal.csd(sig_x, sig_x, Fs, nfft=nfft)
    _, csd_y = signal.csd(sig_y, sig_y, Fs, nfft=nfft)
    _, csd_xy = signal.csd(sig_x, sig_y, Fs, nfft=nfft)
    coh = np.abs(csd_xy)**2 / (csd_x * csd_y)
    if isinstance(f, float):
        coh_f = coh[np.nanargmin(np.abs(f - fqall))]
        fq_diff = np.nanmin(np.abs(f - fqall))
    else:
        coh_f = [coh[np.nanargmin(np.abs(fq - fqall))] for fq in f]
        fq_diff = [np.nanmin(np.abs(fq - fqall)) for fq in f]
    return coh_f, fq_diff


# Correlation utils
def numpy_xcorr(a, b):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, 'full')
    return c


def crosscorr(ssc_a, ssc_b, lag_window):
    rs_np = numpy_xcorr(ssc_a, ssc_b)
    a_lags = np.arange(ssc_a.shape[0] - lag_window, ssc_a.shape[0] + lag_window)
    rs = rs_np[a_lags]
    return rs