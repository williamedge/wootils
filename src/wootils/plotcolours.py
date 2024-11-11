import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
import colorstamps as cs
import matplotlib as mpl


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


def get_quivers_gridded(x_points, y_points, u_data, v_data, thin=[4,4], offset=2):
    X, Y = np.meshgrid(x_points[offset::thin[0]], y_points[offset::thin[1]])
    if u_data.ndim == 2:
        U = u_data[offset::thin[0], offset::thin[1]]
        V = v_data[offset::thin[0], offset::thin[1]]
    elif u_data.ndim == 3:
        U = u_data[:,offset::thin[0], offset::thin[1]]
        V = v_data[:,offset::thin[0], offset::thin[1]]
    return X, Y, U, V

def get_2Dcmap_gridded(u_data, v_data, u_bound, v_bound, stampcmap='peak', **cs_kwargs):
    rgb, stamp = cs.apply_stamp(u_data.values.flatten(),
                                v_data.values.flatten(),
                                stampcmap,
                                vmin_0=v_bound[0], vmax_0=v_bound[1],
                                vmin_1=u_bound[0], vmax_1=u_bound[1],
                                **cs_kwargs)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mymap', rgb, N=len(rgb))
    return cmap, stamp


def cmap_ax(ax, stamp, xticks, yticks, position=[-0.09,0.88], width=0.2, xlabel='U', ylabel='V'):
    cmap_ax = stamp.overlay_ax(ax, lower_left_corner=position, width=width)
    cmap_ax.set_yticks(yticks)
    cmap_ax.set_xticks(xticks)
    cmap_ax.xaxis.tick_top()
    cmap_ax.xaxis.set_label_position('top')
    cmap_ax.set_xlabel(xlabel)
    cmap_ax.set_ylabel(ylabel, rotation=0)
    cmap_ax.text(0.1, 0.9, r'$\nwarrow$', transform=cmap_ax.transAxes, ha='center', va='center')
    cmap_ax.text(0.1, 0.1, r'$\swarrow$', transform=cmap_ax.transAxes, ha='center', va='center')
    cmap_ax.text(0.9, 0.9, r'$\nearrow$', transform=cmap_ax.transAxes, ha='center', va='center')
    cmap_ax.text(0.9, 0.1, r'$\searrow$', transform=cmap_ax.transAxes, ha='center', va='center')
    return cmap_ax



def figure_template(stamp, u_ticks=0.3, v_ticks=0.3, figsize=(8,8)):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    # Add a new axis for the 2D colormap
    overlaid_ax = stamp.overlay_ax(ax, lower_left_corner=[-0.09,0.88], width=0.2)
    overlaid_ax.set_xticks(u_ticks)
    overlaid_ax.set_yticks(v_ticks)
    # Set xticks and label to top
    overlaid_ax.xaxis.tick_top()
    overlaid_ax.xaxis.set_label_position('top')
    overlaid_ax.set_xlabel('U')
    overlaid_ax.set_ylabel('V', rotation=0)
    # Add some text to the colormap
    overlaid_ax.text(0.1, 0.9, r'$\nwarrow$', transform=overlaid_ax.transAxes, ha='center', va='center')
    overlaid_ax.text(0.1, 0.1, r'$\swarrow$', transform=overlaid_ax.transAxes, ha='center', va='center')
    overlaid_ax.text(0.9, 0.9, r'$\nearrow$', transform=overlaid_ax.transAxes, ha='center', va='center')
    overlaid_ax.text(0.9, 0.1, r'$\searrow$', transform=overlaid_ax.transAxes, ha='center', va='center')
    return fig, ax, overlaid_ax


def get_cmap(u_data, v_data, u_bound, v_bound, stampcmap='peak'):
    rgb, stamp = cs.apply_stamp(v_data,
                                u_data,
                                stampcmap,
                                vmin_0=v_bound[0], vmax_0=v_bound[1],
                                vmin_1=u_bound[0], vmax_1=u_bound[1])
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mymap', rgb, N=len(rgb))
    return cmap, stamp


def plot_prelude(ax, prelude=(123.3744, -13.74942), c='orange', s=100, label='F Block', zorder=10):
    ax.scatter(prelude[0], prelude[1], color=c, edgecolors='k', s=s, label=label, zorder=zorder)
    return prelude


def colors_and_quivers(ds, coords=['longitude','latitude'], pvars=['east_vel','north_vel'],\
                       qkwargs={'thin':[1,1], 'offset':0},\
                       cmkwargs={'u_bound':None, 'v_bound':None, 'stampcmap':'peak'}):
    
    Xg, Yg, Uq, Vq = get_quivers_gridded(ds[coords[0]], ds[coords[1]],\
                                            ds[pvars[0]], ds[pvars[1]],\
                                            **qkwargs)
    
    if cmkwargs['u_bound'] is None:
        ll = np.nanmax(np.abs(Uq))
        cmkwargs['u_bound'] = [-ll, ll]
    if cmkwargs['v_bound'] is None:
        ll = np.nanmax(np.abs(Vq))
        cmkwargs['v_bound'] = [-ll, ll]

    # cmap, stamp = get_2Dcmap_gridded(ds[pvars[0]], ds[pvars[1]], **cmkwargs)
    cmap, stamp = get_cmap(ds[pvars[0]].values.flatten(), ds[pvars[1]].values.flatten(),\
                           cmkwargs['u_bound'], cmkwargs['v_bound'], stampcmap=cmkwargs['stampcmap'])
    return Xg, Yg, Uq, Vq, cmap, stamp


def plot_colors_and_quivers(ax, Xg, Yg, Xq, Yq, Uq, Vq, cmap, bathy, qv_dict):
    z = np.arange(np.prod(Xg.shape)).reshape(Xg.shape)
    ax.pcolormesh(Xg, Yg, z, cmap=cmap, shading='auto')
    ax.quiver(Xq, Yq, Uq, Vq, color=[0.3, 0.3, 0.3, 0.25], angles='uv',
              scale=qv_dict['qv_scale'], scale_units='inches', width=qv_dict['qv_width'],
              headwidth=qv_dict['qv_headwidth'], headlength=qv_dict['qv_headlength'])
    ax.tricontour(bathy.xv, bathy.yv, bathy.dv, levels=np.arange(200,1000,200), colors='grey', linewidths=1, linestyles='--')
    ax.set(xlabel='', ylabel='', title='')
