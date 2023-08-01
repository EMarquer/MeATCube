from ..meatcubecb import MeATCubeCB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as plc
import matplotlib.colors as plcol
from typing import Union, Literal
from scipy.interpolate import griddata
import seaborn as sns
from sklearn.decomposition import PCA

def underline_point(X, ax: plt.Axes):
    ax.scatter((X[0]), (X[1]), s=140, linewidths=3, facecolors='none', edgecolors='r')

def influencemap(cb: MeATCubeCB, X, y, case: Union[Literal["best", "worst"], int]="best", ax: plt.Axes=None, transform=None, resolution=50, cmap=None, underline=True):
    """Draws the influence map of one case on all test cases."""

    # Get transformed data and setup a transform if necessary
    if transform is None:
        transform = PCA(n_components=2)
        transform.fit(cb.CB_source)
    X_transformed = transform.transform(X)

    # Get limits
    CB_X_transformed = transform.transform(cb.CB_source)
    if ax is None:
        ax = plt.gca()
        (x_min, x_max) = CB_X_transformed[:,0].min(), CB_X_transformed[:,0].max()
        (y_min, y_max) = CB_X_transformed[:,1].min(), CB_X_transformed[:,1].max()

        # add a small margin
        x_scale = -(x_min - x_max)
        y_scale = -(y_min - y_max)
        (x_min, x_max) = (x_min - (.05 * x_scale), x_max + (.05 * x_scale))
        (y_min, y_max) = (y_min - (.05 * y_scale), y_max + (.05 * y_scale))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        (x_min, x_max) = ax.get_xlim()
        (y_min, y_max) = ax.get_ylim()
    if cmap is None:
        cmap = sns.color_palette("vlag", as_cmap=True)
        cmap = plcol.ListedColormap(cmap.colors[::-1])

    # with initial case base,
    # draw two influence maps for less and most competent case
    # compute the influence of each source case on the test set
    influences = cb.influence(X, y, aggregation=None) # [|CB|, |X|]
    # compute the competence of each source case on the test set
    case_comp = influences.mean(dim=-1) # [|CB|]

    if case == "best":
        title = "Best case"
    elif case == "worst":
        title = "Worst case"
    else:
        title = f"Case {case}"

    if isinstance(case, int):
        influence = cb.competence(X, y, index=case, aggregation=None) # [|X|]
        case_comp = influence.mean(dim=-1) # []
    else:
        # compute the influence of each source case on the test set
        influences = cb.influence(X, y, aggregation=None) # [|CB|, |X|]
        # compute the competence of each source case on the test set
        cases_comp = influences.mean(dim=-1) # [|CB|]
        # which source case is the worse? the best?
        if case == "best":
            case = cases_comp.argmax()
        elif case == "worst":
            case = cases_comp.argmin()
        influence = influences[case]
        case_comp = cases_comp[case]
    title += f" (case competence {case_comp:.3f})"

    # now draw the maps using a same scale
    bound = max(abs(influence.min()), influence.max())

    ax.set_title(title)
    if underline: underline_point(CB_X_transformed[case], ax)

    
    # create a mesh of points
    grid_x, grid_y = np.mgrid[x_min:x_max:complex(resolution),y_min:y_max:complex(resolution)]
    z = griddata(X_transformed, influence, (grid_x, grid_y), method='cubic').T

    return ax.imshow(z, cmap=cmap, vmin=-bound, vmax=bound, extent=(x_min, x_max, y_min, y_max), origin="lower", alpha=0.35), transform
