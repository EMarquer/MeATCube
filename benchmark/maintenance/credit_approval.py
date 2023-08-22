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
EUCLIDEAN_SIM = lambda x,y: np.exp(- np.linalg.norm(x.astype(float) - y.astype(float)))
EQUALITY_SIM = lambda x,y: (True if x == y else False)
#def EQUALITY_SIM(x,y): print (x, y); return (True if x == y else False)
def euclidean_equality_mix(euclidean_columns, equality_columns):
    """Given some categorical and continuous attributes, generates a similarity"""
    sim = lambda x,y: (
        EUCLIDEAN_SIM(x[euclidean_columns], y[euclidean_columns])*len(euclidean_columns) + 
        sum(EQUALITY_SIM(x[c], y[c]) for c in equality_columns)
        )/(len(euclidean_columns) + len(equality_columns))
    return sim
# Columns in this dataset:
# A1:	b, a.
# A2:	continuous.
# A3:	continuous.
# A4:	u, y, l, t.
# A5:	g, p, gg.
# A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
# A7:	v, h, bb, j, n, z, dd, ff, o.
# A8:	continuous.
# A9:	t, f.
# A10:	t, f.
# A11:	continuous.
# A12:	t, f.
# A13:	g, p, s.
# A14:	continuous.
# A15:	continuous.
# A16: +,-         (class attribute)

# global definitions
file = os.path.join(THIS_FOLDER, "datasets", "credit+approval", "crx.data")
columns = [f"A{n}" for n in range (1,17)]
target_column = "A16"
source_similarity = euclidean_equality_mix([n-1 for n in [2, 3, 8, 11, 14, 15]], [n-1 for n in [1, 4, 5, 6, 7, 9, 10, 12, 13]])
outcome_similarity = EQUALITY_SIM
dataset = "Credit (credit+approval)"

# load the data
df = pd.read_csv(file, header=None)
df.columns = columns
# drop values that are not OK (cf https://cora.ucc.ie/server/api/core/bitstreams/39193798-3fe0-461a-b1b6-3d9cffd108d3/content)
# we need to remove 37 datapoints containing unknown values ("?")
to_remove = (
    sum([(df[f"A{n}"] == "?") for n in range(1,17)])
).astype(bool)
df=df[~to_remove]

X = df[[c for c in columns if c!=target_column]]
y = df[target_column]
y_values = np.unique(y)
print(X.iloc[0])

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
