import numpy as np
import matplotlib.pyplot as plt


def add_warnbars(time, warn_idx, ax):  
    # Loop thorugh warnings and axes
    for txx in time[warn_idx].values:
        for x in ax:
            # Get plot bounds
            x.axvline(txx, ymin=0, ymax=0.1, c='k', lw=1)
            x.axvline(txx, ymin=0.9, ymax=1., c='k', lw=1)