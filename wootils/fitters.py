import numpy as np


def murphy_ss(true_signal, fit_signal):
    return 1 - (np.nanmean((true_signal - fit_signal)**2) / np.nanmean((true_signal - np.nanmean(fit_signal))**2))

