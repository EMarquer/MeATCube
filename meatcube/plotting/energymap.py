from ..meatcubecb import MeATCubeCB
from ..metrics import confidence
from typing import List, Tuple, Any
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.collections as plc
import matplotlib.colors as plcol

def energymap_points(X, transform=None, ax: plt.Axes=None, resolution=50):
    """Given some points (numerical data only) generates the necessary 
    coordinates to cover the figure.
    
    :param transform: if `transform` is provided, it should support `.transform`
    and `.inverse_transform`; if `transform` is `None`, a PCA will be learned 
    from X (requires a value of X suitable for `PCA(n_components=2).fit(X)`)

    :param ax: if an `ax` is provided, use its current boundaries to define the
    coordinates to cover; if `ax` is `None`, the transformed values of X will be
    used to determine the area (min and max for each dimension, with an added 
    margin of 5% in each direction; requires a value of X suitable for 
    `transform.transform(X)`)

    :param resolution: the number of points to take in each dimension of the 
    plot
    """

    # define a transform
    if transform is None:
        transform = PCA(n_components=2)
        transform.fit(X)

    if ax is None:
        X_transformed = transform.transform(X)
        (x_min, x_max) = X_transformed[:,0].min(), X_transformed[:,0].max()
        (y_min, y_max) = X_transformed[:,1].min(), X_transformed[:,1].max()
        
        # add a small margin
        x_scale = -(x_min - x_max)
        y_scale = -(y_min - y_max)
        (x_min, x_max) = (x_min - (.05 * x_scale), x_max + (.05 * x_scale))
        (y_min, y_max) = (y_min - (.05 * y_scale), y_max + (.05 * y_scale))

        x_step = x_scale/resolution
        y_step = y_scale/resolution
    else:
        (x_min, x_max) = ax.get_xlim()
        (y_min, y_max) = ax.get_ylim()

        x_step = (x_max-x_min)/resolution
        y_step = (y_max-y_min)/resolution
    
    # create a mesh of points covering the space
    xk = np.arange(x_min-(.5+round(resolution/10))*x_step,
                   x_max+((2+round(resolution/10))*x_step), x_step)
    yk = np.arange(y_min-(.5+round(resolution/10))*y_step,
                   y_max+((2+round(resolution/10))*y_step), y_step)
    X_2D_heatmap = np.array(list(product(xk, yk)))
    X_heatmap = transform.inverse_transform(X_2D_heatmap)

    return xk, yk, X_2D_heatmap, X_heatmap, transform, (x_min, x_max), (y_min, y_max)

def fill_energymap_values(cb: MeATCubeCB, xk, yk, X_2D_heatmap, X_heatmap, batched=False):
    # get the confidence
    conf = confidence(cb, X_heatmap, batched=batched)

    # normalize to get conf as alpha
    max_conf = conf.max()
    min_conf = conf.min()
    alpha = (conf - min_conf) / (max_conf - min_conf)

    # transform the confidence results into a 2D confidence map
    z = np.zeros((len(cb.potential_outcomes), len(yk), len(xk)))
    for outcome_id in range(len(cb.potential_outcomes)):
        for (x_i, x_j), conf_ij, (i, j) in zip(X_2D_heatmap, alpha[outcome_id], product(range(len(xk)), range(len(yk)))):
            z[outcome_id, j, i] = conf_ij
    return z

def energymap(cb: MeATCubeCB, X=None, transform=None, outcome_colors=None, ax: plt.Axes=None, resolution=50, alpha=.35, batched=False) -> Tuple[List[plc.QuadMesh], Any]:
    if outcome_colors is None: outcome_colors = sns.color_palette()
    if X is None: X = cb.CB_source

    # generate points
    xk, yk, X_2D_heatmap, X_heatmap, transform, (x_min, x_max), (y_min, y_max) = energymap_points(
        X, transform=transform, ax=ax, resolution=resolution)
    
    # generate an ax if necessary
    if ax is None:
        ax = plt.gca()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # find values for the points
    z = fill_energymap_values(cb, xk, yk, X_2D_heatmap, X_heatmap, batched=batched)

    # display the heatmap of each class
    energy_maps = []
    for outcome_id in range(len(cb.potential_outcomes)):
        cmap = rgb2alpha(outcome_colors[outcome_id], max_alpha=alpha)
        energy_maps.append(
            ax.pcolormesh(xk, yk, z[outcome_id], cmap=cmap,
                  shading='nearest', vmin=0, vmax=1 # already normalized
        ))
    return energy_maps, transform

def update_energymap(cb: MeATCubeCB, heatmap: List[plc.QuadMesh], transform, resolution=50):
    ax = heatmap[0].axes
    # generate points
    xk, yk, X_2D_heatmap, X_heatmap, transform, (x_min, x_max), (y_min, y_max) = energymap_points(
        None, transform=transform, ax=ax, resolution=resolution)

    # find values for the points
    z = fill_energymap_values(cb, xk, yk, X_2D_heatmap, X_heatmap)

    for outcome_id in range(len(cb.potential_outcomes)):
        heatmap[outcome_id].set_array(z[outcome_id])

def rgb2alpha(rgb, n_steps=20, max_alpha=1):
    rgb = np.repeat([rgb], n_steps, axis=0)
    alpha = np.expand_dims(np.linspace(0, max_alpha, n_steps), axis=-1)
    rgba_lin = np.concatenate([rgb, alpha], axis=-1)
    cmap = plcol.ListedColormap(rgba_lin)
    return cmap