import numpy as np


def halfnormal_std(x):
    return np.sqrt(np.sum(2*x**2) / (2*len(x)))

def halfnormal_nanstd(x):
    return np.sqrt(np.nansum(2*x**2) / (2*len(x)))