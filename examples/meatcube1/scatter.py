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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=42, stratify=y)

# add root directory to be able to import MeATCube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import meatcube as mc

# create the CB
source_similarity = lambda x,y: np.exp(- np.linalg.norm(x - y))
outcome_similarity = lambda x,y: (True if x == y else False)
cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)

# plot the best and worst case
from meatcube.plotting.cbscatter import plot_cb, plot_ref
import matplotlib.pyplot as plt

# Ex 1: the transform is prepared for cases in the CB and reused on the references
_, transform = plot_cb(cb)
plot_ref(cb, X_test, y_test, transform=transform)
plt.show()
plt.clf()

# Ex 2: so that the transform considers both the train and test set
transform = PCA(2).fit(X, y)
plot_cb(cb, transform=transform)
plot_ref(cb, X_test, y_test, transform=transform)
plt.show()
plt.clf()
# %%
