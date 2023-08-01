# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
import pandas as pd
import numpy as np

USE_STRING_VALUES = True

iris = load_iris(as_frame=True)

X: pd.DataFrame = iris["data"] # source
y = iris["target"] # target

if USE_STRING_VALUES:
    # to test with strings as labels
    y = y.apply(lambda x: iris["target_names"][x]) 
    y_values = iris["target_names"]
else:
    y_values = np.unique(y)

# stratified splitting of the data (take only 8 cases to have incompetent enough cases)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=142, random_state=42, stratify=y)

# add root directory to be able to import MeATCube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import meatcube as mc

# create the CB
source_similarity = lambda x,y: np.exp(- np.linalg.norm(x - y))
outcome_similarity = lambda x,y: (True if x == y else False)
cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)

# plot the best and worst case
from meatcube.plotting.preprocessing import prepare_ax
from meatcube.plotting.influencemap import influencemap
import matplotlib.pyplot as plt

# Ex 1: influence of the most competent case from the CB
ax, transform = prepare_ax(cb, X)
influencemap(cb, X_test, y_test, transform=transform, ax=ax)
plt.show()
plt.clf()

# Ex 2: influence of the least competent case from the CB
ax, transform = prepare_ax(cb, X)
influencemap(cb, X_test, y_test, case="worst", transform=transform, ax=ax)
plt.show()
plt.clf()

# Ex 3: influence of an arbitrary case from the CB, and plot the cb on top
from meatcube.plotting.cbscatter import plot_cb, plot_ref
ax, transform = prepare_ax(cb, X)
influencemap(cb, X_test, y_test, case=2, transform=transform, ax=ax)
plot_cb(cb, alpha=0.5, transform=transform, ax=ax)
plot_ref(cb, X_test, y_test, cb.predict(X_test), alpha=0.5, transform=transform, ax=ax)
plt.show()
plt.clf()
# %%
