from ..meatcubecb import MeATCubeCB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as plc
import matplotlib.colors as plcol
from typing import Union, Literal
from scipy.interpolate import griddata
import seaborn as sns
from sklearn.decomposition import PCA

MISTAKES_AS_PRED_COLOR = True
CORRECT_SCATTER_KWARGS = {"marker": "v", "label": "Success"}
MISTAKE_SCATTER_KWARGS = {"marker": "X", "label": "Failure"}

def to_c_cb(cb: MeATCubeCB, y, outcome_colors=None):
    if outcome_colors is None: outcome_colors = sns.color_palette()

    return [outcome_colors[y_id] for y_id in cb.outcome_index(y)]
def to_c_ref(cb: MeATCubeCB, y, outcome_colors=None, color_factor=1.5):
    if outcome_colors is None: outcome_colors = sns.color_palette()

    return [np.clip(np.array(outcome_colors[y_id]) * color_factor, 0, 1) for y_id in cb.outcome_index(y)]

def plot_cb(cb: MeATCubeCB, ax: plt.Axes=None, transform=None, outcome_colors=None, alpha=1):
    """Plots the cases in the CB as scattered points."""

    # Get transformed data and setup a transform if necessary
    if transform is None:
        transform = PCA(n_components=2)
        transform.fit(cb.CB_source)
    X_transformed = transform.transform(cb.CB_source)

    if ax is None:
        ax = plt.gca()

    # the actual plotting part
    path_col = ax.scatter(X_transformed[:,0], X_transformed[:,1], c=to_c_cb(cb, cb.CB_outcome), alpha=alpha)
    return path_col, transform

def plot_ref(cb: MeATCubeCB, X, y, y_pred=None, ax: plt.Axes=None, transform=None, outcome_colors=None, alpha=1):
    """Plots the reference cases as scattered points.
    
    :param y_pred: 
        if True or an array of predictions is given, will change the shape of the dots to a triangle (pred=gold) or a cross (pred!=gold)cb
        if True, will do the prediction internally
    """

    # Get transformed data and setup a transform if necessary
    if transform is None:
        transform = PCA(n_components=2)
        transform.fit(X)
    X_transformed = transform.transform(X)

    if ax is None:
        ax = plt.gca()

    # the actual plotting part
    if y_pred is True:
        y_pred = cb.predict(X)
        
    if y_pred is None:
        path_col = ax.scatter(X_transformed[:,0], X_transformed[:,1], c=to_c_ref(cb, y), alpha=alpha)
    else:
        correct_mask = y == y_pred
        X_correct, X_mistake = X_transformed[correct_mask], X_transformed[~correct_mask]
        y_correct, y_mistake = y[correct_mask], y[~correct_mask]
        y_pred_mistake = y_pred[~correct_mask]

        path_col = [
            ax.scatter(X_correct[:,0], X_correct[:,1], c=to_c_ref(cb, y_correct), **CORRECT_SCATTER_KWARGS, alpha=alpha),
            ax.scatter(X_mistake[:,0], X_mistake[:,1], c=to_c_ref(cb, y_pred_mistake if MISTAKES_AS_PRED_COLOR else y_mistake), **MISTAKE_SCATTER_KWARGS, alpha=alpha),
        ]
    return path_col, transform