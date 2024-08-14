import numpy as np
import string
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def save(fname, fig, format='png', transparent=True, dpi=300):
    fig.savefig(fname + '.' + format, format=format, transparent=transparent,\
                bbox_inches='tight', pad_inches=0.06, dpi=dpi)
    return fig

def saveclose(fname, fig):
    save(fname, fig)
    plt.close(fig)

def saveagu(fname, fig):
    save(fname, fig, format='jpg', transparent=False, dpi=600)

def plot_align(ax, axspec=None):
    if axspec is None:
        x_lft = np.min([x.get_position().bounds[0] for x in ax])
        x_rgt = np.min([x.get_position().bounds[2] for x in ax])
    else:
        x_lft = axspec.get_position().bounds[0]
        x_rgt = axspec.get_position().bounds[2]
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


def basic_ts(time, ax, xlim=None, strip_titles=True, unrotate_xlabs=False):
    if xlim is None:
        xlim = (time[0], time[-1])
    # try:
    #     lx = len(ax)
    # except:
    #     ax = [ax]
    if np.size(ax) == 1:
        ax = [ax]
    for x in ax:
        x.set_xlim(xlim)
        # x.grid()
        x.set_xlabel('')
        if strip_titles:
            x.set_title('')
        if unrotate_xlabs:
            x.set_xticklabels(x.get_xticklabels(), rotation=0, ha="center")
        if x != ax[-1]:
            x.set_xticklabels('')
    return None


def plot_axislabels(ax, pos='topleft', h_ratios=None, type='a', c='k', xpos=None, ypos=None, fs=12, weight='bold'):
    if not xpos:
        if pos == 'topleft':
            xpos = 0.03
        elif pos == 'topright':
            xpos = 0.97
        else:
            raise ValueError('pos must be topleft or topright')
    if not ypos:
        if h_ratios:
            h_ratios = np.array(h_ratios)
            ypos = 0.75 + 0.03*(h_ratios - 1)
        else:
            ypos = np.repeat(0.85, np.size(ax))
    if type == 'a':
        # Create a list of letters of len(ax)
        labels = list(string.ascii_lowercase)[:len(ax)]
    elif type == 'A':
        labels = list(string.ascii_uppercase)[:len(ax)]
    elif type == '1':
        labels = list(string.digits)[:len(ax)]
    else:
        raise ValueError('type must be a, A or 1')
    if np.size(ax) == 1:
        ax = [ax]
    # Add the labels
    for x,l, yp in zip(ax, labels, ypos):
        x.text(xpos, yp, '(' + l + ')', c=c, horizontalalignment='center', verticalalignment='center',\
               transform=x.transAxes, weight=weight, fontsize=fs)
    return None


def set_mirrorlim(ax, data, which='y'):
    lim_plt = np.max(np.abs(data))
    if which == 'y':
        ax.set_ylim(-1*lim_plt, lim_plt)
    elif which == 'x':
        ax.set_xlim(-1*lim_plt, lim_plt)