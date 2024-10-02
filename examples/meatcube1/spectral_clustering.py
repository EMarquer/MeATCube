# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

USE_STRING_VALUES = False

# iris = load_iris(as_frame=True)

# X: pd.DataFrame = iris["data"] # source
# y = iris["target"] # target

# if USE_STRING_VALUES:
#     # to test with strings as labels
#     y = y.apply(lambda x: iris["target_names"][x]) 
#     y_values = iris["target_names"]
# else:
#     y_values = np.unique(y)
import sys, os
df: pd.DataFrame = pd.read_csv(
    os.path.join(os.path.dirname(__file__), '..', 'example_data', "doughnut_25_50_5_1000.csv"),
    header=None, index_col=None)
df.columns = ["x", "y", "target"]
X: pd.DataFrame = df[["x", "y"]] # source
y = df["target"] # target
y_values = np.unique(y)
# %%

# stratified splitting of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, train_size=50, random_state=42, stratify=y)


# add root directory to be able to import MeATCube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import meatcube as mc

# create the CB
source_similarity = lambda x,y: np.exp(- np.linalg.norm(x - y))
outcome_similarity = lambda x,y: (True if x == y else False)
cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)

# %%
### Hinge Competence (default) ###
# competence of the cases


# %%
import meatcube.plotting as mcp
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import matplotlib.cm as pltcm
from sklearn.cluster import SpectralClustering
def plot_ranking(cb: mc.MeATCubeCB, X_test, y_test, rank_indices = slice(10), color="rank", ax=None, colorbar=True):
    cb.compute_sim_matrix()
    
    clustering_source = SpectralClustering(affinity="precomputed").fit_predict(cb.source_sim_matrix)
    clustering_outcome = SpectralClustering(affinity="precomputed").fit_predict(cb.outcome_sim_matrix)

    comp = cb.case_competences(X_test, y_test)
    ranks=comp.argsort(dim=-1)
    print(f"Competence of the CB: {comp.mean()}")
    
    ax, transform = mcp.prepare_ax(cb, X, ax=ax)
    mcp.energymap(cb, ax=ax, transform=transform)
    mcp.plot_cb(cb, ax=ax, transform=transform)
    mcp.plot_ref(cb, X_test, y_test, True, ax=ax, transform=transform, alpha=0.3)
    points=transform.transform(cb.CB_source)
    points=points[ranks]
    comp=comp[ranks]
    clustering_source=clustering_source[ranks]
    clustering_outcome=clustering_outcome[ranks]
    # if color=="rank":
    #     values=np.arange(ranks.size(0))
    # elif color=="competence":
    #     values=comp
    # else:
    #     raise ValueError("Choose between `rank` and `competence` for the `color` kwarg.")
    values=clustering_source

    color_mappable=pltcm.ScalarMappable(norm=pltc.Normalize(values.min(), values.max()), cmap="RdYlGn")
    c = [color_mappable.to_rgba(value) for value in values]
    ax.scatter(points[:,0],
               points[:,1],
               edgecolors=c,
               s=100,
               linewidths=3,
               facecolors='none',
               alpha=0.5)
    #plt.colorbar(ax=ax)

    rank_indices = slice(10)
    for (x,y_), label in zip(points[rank_indices], np.arange(ranks.size(0))[rank_indices]):
        ax.text(x, y_, label.item(), ha='center', va='top')

    if colorbar: plt.colorbar(color_mappable, ax=ax, label=color.capitalize())
    return comp.mean().item()
plot_ranking(cb, X_test, y_test)

# %%
#Only the top 10

ncols=20
scale=4
step_sizes=[2, 3, 6]
nrows=len(step_sizes)+1
plt.figure(figsize=(ncols*scale, nrows*scale))
ax=plt.subplot(nrows, ncols, 1)
ax.set_title("Step 0")
plot_ranking(cb, X_test, y_test, ax=ax, colorbar=False)
cb_=cb
cb_1=cb
import meatcube.maintenance as mcm
from meatcube.metrics import f1_score
# step size 1
for i in range(1, ncols):
    cb_1=mcm.decrement(cb_1, X_test, y_test,return_all=False,k=1)
    
    print(f"== Step {(i)} (CB size {len(cb_1)}) ==")
    ax=plt.subplot(nrows, ncols, 1+i)
    comp = plot_ranking(cb_1, X_test, y_test, ax=ax, colorbar=False)
    f1=f1_score(cb_1, X_test, y_test)
    ax.set_title(f"Step {(i)} (CB size {len(cb_1)})\nC={comp:3f}, F1={f1:3f})")

for j, step_size in enumerate(step_sizes):
    cb_=cb
    ax=plt.subplot(nrows, ncols, (ncols*(j+1))+1)
    comp = plot_ranking(cb_, X_test, y_test, ax=ax, colorbar=False)

    for i in range(step_size, ncols, step_size):
        ax=plt.subplot(nrows, ncols, (ncols*(j+1))+i+1)
        cb_=mcm.decrement(cb_, X_test, y_test,return_all=False,k=step_size)
        
        print(f"== Step {i} (CB size {len(cb_)}) ==")
        comp = plot_ranking(cb_, X_test, y_test, ax=ax, colorbar=False)
        f1=f1_score(cb_, X_test, y_test)
        ax.set_title(f"Step {i}: C={comp:.3f}, F1={f1:3f}")
# %%
