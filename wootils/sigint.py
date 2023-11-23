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