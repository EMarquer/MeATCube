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
DATASETS = {
    "Balance": {
        "file": os.path.join(THIS_FOLDER, "datasets", "balance+scale", "balance-scale.data"),
        "columns": ["Class Name", "Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"],
        "target column": "Class Name"
    }
}

for dataset in DATASETS.keys():
    dataset = "Balance"
    df = pd.read_csv(DATASETS[dataset]["file"], header=None)
    df.columns = DATASETS[dataset]["columns"]
    X = df[[c for c in DATASETS[dataset]["columns"] if c!=DATASETS[dataset]["target column"]]]
    y = df[DATASETS[dataset]["target column"]]
    y_values = np.unique(y)

    # 10 splits of 60%, 20%, with 20% test set
    from sklearn.model_selection import KFold, train_test_split
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=.25, random_state=42)

    # create the CB
    source_similarity = lambda x,y: np.exp(- np.linalg.norm(x - y))
    outcome_similarity = lambda x,y: (True if x == y else False)
    cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)
    cb.compute_sim_matrix()

    cb_ = decrement(cb, X_dev, y_dev, return_all=False)

# %%
    # compress
    cb_ = decrement_early_stopping(cb, X_dev, y_dev, register="accuracy", monitor="accuracy", return_all=False)

    # evaluate
    acc = accuracy(cb, X_test, y_test, average="macro")
    acc_ = accuracy(cb_, X_test, y_test, average="macro")
    print(f"{dataset} dataset --- \tAccuracy initial: {acc:%}\tAccuracy final: {acc_:%}\tDeletion rate: {len(cb_)/len(cb):%} cases in the CB, {X_test.shape[0]} cases in the test set)")

# %%
