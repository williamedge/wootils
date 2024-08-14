import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def coloured_line(x_data, y_data, c_data, cmap='Blues', cmin=0.0, cmax=1.0):
    points = np.array([x_data, y_data]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(np.nanpercentile(c_data, cmin),\
                        np.nanpercentile(c_data, 100*cmax))
    cmap = plt.get_cmap(cmap)
    new_cmap = truncate_colormap(cmap, cmin, cmax)
    lc = LineCollection(segments, cmap=new_cmap, norm=norm)
    lc.set_array(c_data)
    return lc