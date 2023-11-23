import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def save(fname, fig, format='png', transparent=True, dpi=300):
    fig.savefig(fname + '.' + format, format=format, transparent=transparent,\
                bbox_inches='tight', pad_inches=0.01, dpi=dpi)
    return fig

def saveclose(fname, fig):
    save(fname, fig)
    plt.close(fig)

def saveagu(fname, fig):
    save(fname, fig, format='jpg', transparent=False, dpi=600)

def plot_align(ax):
    x_lft = np.min([x.get_position().bounds[0] for x in ax])
    x_rgt = np.min([x.get_position().bounds[2] for x in ax])
    for xx in ax:
        xx.set_position([x_lft, xx.get_position().bounds[1], x_rgt, xx.get_position().bounds[3]])

def vert_stack(subplots, hsize=8, vsize=1.5, hspace=0.05, h_ratio=None, **kwargs):
    if h_ratio is None:
        fig, ax = plt.subplots(subplots,1, figsize=(hsize,vsize*subplots),\
                           gridspec_kw={'hspace':hspace}, **kwargs)
    else:
        fig, ax = plt.subplots(subplots,1, figsize=(hsize,vsize*subplots),\
                           gridspec_kw={'hspace':hspace, 'height_ratios':h_ratio}, **kwargs)
    return fig, ax


def horz_stack(subplots, hsize=8, vsize=2.5, wspace=0.075, w_ratio=None, **kwargs):
    if w_ratio is None:
        fig, ax = plt.subplots(1, subplots, figsize=(hsize, vsize),\
                            gridspec_kw={'wspace':wspace}, **kwargs)
    else:
        fig, ax = plt.subplots(1, subplots, figsize=(hsize, vsize),\
                            gridspec_kw={'wspace':wspace, 'width_ratios':w_ratio}, **kwargs)
    return fig, ax


def basic_ts(time, ax, xlim=None, strip_titles=True):
    if xlim is None:
        xlim = (time[0], time[-1])
    try:
        lx = len(ax)
    except:
        ax = [ax]
    for x in ax:
        x.set_xlim(xlim)
        # x.grid()
        x.set_xlabel('')
        if strip_titles:
            x.set_title('')
        x.set_xticklabels(x.get_xticklabels(), rotation=0, ha="center")
        if x != ax[-1]:
            x.set_xticklabels('')