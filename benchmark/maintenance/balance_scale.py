# %%
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import meatcube as mc
from meatcube.maintenance import decrement_early_stopping, decrement
from meatcube.metrics import accuracy

# refers to https://cora.ucc.ie/server/api/core/bitstreams/39193798-3fe0-461a-b1b6-3d9cffd108d3/content
THIS_FOLDER = os.path.dirname(__file__)
EUCLIDEAN_SIM = lambda x,y: np.exp(- np.linalg.norm(x - y))
EQUALITY_SIM = lambda x,y: (True if x == y else False)
USE_ORDINAL_OUTCOMES = True # significantly better results by just setting this to true

# global definitions
file = os.path.join(THIS_FOLDER, "datasets", "balance+scale", "balance-scale.data")
columns = ["Class Name", "Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"]
target_column = "Class Name"
source_similarity = EUCLIDEAN_SIM
if USE_ORDINAL_OUTCOMES:
    outcome_ordinal_values = {"L": -1, "B": 0, "R": 1}
    outcome_similarity = EUCLIDEAN_SIM
else:
    EQUALITY_SIM
dataset = "Balance (balance+scale)"

# load the data
df = pd.read_csv(file, header=None)
df.columns = columns
X = df[[c for c in columns if c!=target_column]]
y = df[target_column]
if USE_ORDINAL_OUTCOMES: # switch to ordinal
    y = y.apply(lambda v: outcome_ordinal_values[v])
y_values = np.unique(y)

# 10 splits of 60%, 20%, with 20% test set
from sklearn.model_selection import train_test_split
X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=.2, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=.25, random_state=42)

# create the CB
cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)
cb.compute_sim_matrix()

# evaluate
acc = accuracy(cb, X_test, y_test)
print(f"{dataset} dataset --- \tAccuracy initial: {acc:%}")

from datetime import datetime
print(f"starting decrement: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
#cb_ = decrement(cb, X_dev, y_dev, return_all=False, batch_size=16)

## %%
# compress
cb_ = decrement_early_stopping(cb, X_dev, y_dev, register="accuracy", monitor="accuracy", return_all=False, batch_size=8)
print(f"finish decrement: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# evaluate
acc = accuracy(cb, X_test, y_test)
acc_ = accuracy(cb_, X_test, y_test)
print(f"{dataset} dataset --- \tAccuracy initial: {acc:%}\tAccuracy final: {acc_:%}\tDeletion rate: {1-len(cb_)/len(cb):%} cases in the CB, {X_test.shape[0]} cases in the test set)")

# %%
