from ..meatcubecb import MeATCubeCB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def prepare_ax(cb: MeATCubeCB, X=None, ax: plt.Axes=None, transform=None):
    """Draws the influence map of one case on all test cases."""

    if X is None:
        X = cb.CB_source
    else:
        X = np.concatenate([cb.CB_source, X], axis=0)

    # Get transformed data and setup a transform if necessary
    if transform is None:
        transform = PCA(n_components=2)
        transform.fit(X)
    
    # Get limits
    X_transformed = transform.transform(X)
    (x_min, x_max) = X_transformed[:,0].min(), X_transformed[:,0].max()
    (y_min, y_max) = X_transformed[:,1].min(), X_transformed[:,1].max()
    
    # add a small margin
    x_scale = -(x_min - x_max)
    y_scale = -(y_min - y_max)
    (x_min, x_max) = (x_min - (.05 * x_scale), x_max + (.05 * x_scale))
    (y_min, y_max) = (y_min - (.05 * y_scale), y_max + (.05 * y_scale))

    if ax is None:
        ax = plt.gca()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        (x_min, x_max) = ax.get_xlim()
        (y_min, y_max) = ax.get_ylim()
    
    return ax, transform