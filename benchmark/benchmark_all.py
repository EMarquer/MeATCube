# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns



# %% [markdown]
# Models

# %%


# as the code is loaded from a subfolder, we use the following snippet to add meatcube2 to the import path
# for a normal usage with meatcube2 installed, the two following lines are unnecessary
import sys, os
try:
    CURRENT_FOLDER = os.path.dirname(__file__) # normal way
except NameError:
    CURRENT_FOLDER = globals()['_dh'][0] # jupyter notebook way
sys.path.append(sys.path.join(CURRENT_FOLDER, ".."))

# we load meatcube2 
from meatcube2.models import MeATCubeCB, CtCoAT, CtCoATNaive, EnergyKNN, EnergyClf
from meatcube2.cb_maintenance import CBClassificationMaintainer
from sklearn.neighbors import KNeighborsClassifier

from scipy.linalg import norm
from scipy.spatial.distance import euclidean, cosine, correlation, braycurtis, hamming
def euclidean_sim(x1,x2):
    return np.exp(-euclidean(x1,x2))
def cosine_sim(x1,x2):
    return np.exp(-cosine(x1,x2))
def correlation_sim(x1,x2):
    return np.exp(-correlation(x1,x2))
def hamming_sim(x1,x2):
    return np.exp(-hamming(x1,x2))
def radius_sim(x1,x2):
    return np.exp(-abs(norm(x1)-norm(x2)))
def class_equality_sim(y1,y2):
    return np.equal(y1,y2).astype(float)

classifier_names = [
    "MeATCube \n($\sigma_X$: Euclidean, $\sigma_y$: class)",
    # "MeATCube \n($\sigma_X$: Cosine, $\sigma_y$: class)",
    # "MeATCube \n($\sigma_X$: Hamming, $\sigma_y$: class)",
    "CtCoat \n($\sigma_X$: Euclidean, $\sigma_y$: class)",
    # "CtCoat \n($\sigma_X$: Cosine, $\sigma_y$: class)",
    # "CtCoat \n($\sigma_X$: Hamming, $\sigma_y$: class)",
    # "kNN \n($k=1$, sklearn kNN)",
    "kNN \n($k=1$, $\sigma_X$: Euclidean, $\sigma_y$: class)",
    "kNN \n($k=3$, $\sigma_X$: Euclidean, $\sigma_y$: class)",
    "kNN \n($k=10$, $\sigma_X$: Euclidean, $\sigma_y$: class)",
    # "kNN \n($k=1$, $\sigma_X$: Cosine, $\sigma_y$: class)",
    # "kNN \n($k=3$, $\sigma_X$: Cosine, $\sigma_y$: class)",
    # "kNN \n($k=10$, $\sigma_X$: Cosine, $\sigma_y$: class)",
    # "kNN \n($k=1$, $\sigma_X$: Hamming, $\sigma_y$: class)",
    # "kNN \n($k=3$, $\sigma_X$: Hamming, $\sigma_y$: class)",
    # "kNN \n($k=10$, $\sigma_X$: Hamming, $\sigma_y$: class)",
]
classifier_properties = [
    {"model": "MeATCube",       "$\sigma_X$": "Euclidean",  "$\sigma_y$": "class"},
    # {"model": "MeATCube",       "$\sigma_X$": "Cosine",     "$\sigma_y$": "class"},
    # {"model": "MeATCube",       "$\sigma_X$": "Hamming",    "$\sigma_y$": "class"},
    {"model": "CtCoat",         "$\sigma_X$": "Euclidean",  "$\sigma_y$": "class"},
    # {"model": "CtCoat",         "$\sigma_X$": "Cosine",     "$\sigma_y$": "class"},
    # {"model": "CtCoat",         "$\sigma_X$": "Hamming",    "$\sigma_y$": "class"},
    # {"model": "sklearn kNN", "k": 1},
    {"model": "kNN", "k": 1,    "$\sigma_X$": "Euclidean",  "$\sigma_y$": "class"},
    {"model": "kNN", "k": 3,    "$\sigma_X$": "Euclidean",  "$\sigma_y$": "class"},
    {"model": "kNN", "k": 10,   "$\sigma_X$": "Euclidean",  "$\sigma_y$": "class"},
    # {"model": "kNN", "k": 1,    "$\sigma_X$": "Cosine",     "$\sigma_y$": "class"},
    # {"model": "kNN", "k": 3,    "$\sigma_X$": "Cosine",     "$\sigma_y$": "class"},
    # {"model": "kNN", "k": 10,   "$\sigma_X$": "Cosine",     "$\sigma_y$": "class"},
    # {"model": "kNN", "k": 1,    "$\sigma_X$": "Hamming",    "$\sigma_y$": "class"},
    # {"model": "kNN", "k": 3,    "$\sigma_X$": "Hamming",    "$\sigma_y$": "class"},
    # {"model": "kNN", "k": 10,   "$\sigma_X$": "Hamming",    "$\sigma_y$": "class"},
]

classifiers = [
    MeATCubeCB(euclidean_sim, class_equality_sim, precompute_sim_matrix=True),
    # MeATCubeCB(cosine_sim,    class_equality_sim, precompute_sim_matrix=True),
    # MeATCubeCB(hamming_sim,   class_equality_sim, precompute_sim_matrix=True),
    CtCoAT(euclidean_sim, class_equality_sim, precompute_sim_matrix=True),
    # CtCoAT(cosine_sim,    class_equality_sim, precompute_sim_matrix=True),
    # CtCoAT(hamming_sim,   class_equality_sim, precompute_sim_matrix=True),
    EnergyClf(KNeighborsClassifier(n_neighbors=1, n_jobs=8)),
    EnergyKNN(euclidean_sim, class_equality_sim, n_neighbors=1,  precompute_sim_matrix=True),
    EnergyKNN(euclidean_sim, class_equality_sim, n_neighbors=3,  precompute_sim_matrix=True),
    EnergyKNN(euclidean_sim, class_equality_sim, n_neighbors=10, precompute_sim_matrix=True),
    # EnergyKNN(cosine_sim,    class_equality_sim, n_neighbors=1,  precompute_sim_matrix=True),
    # EnergyKNN(cosine_sim,    class_equality_sim, n_neighbors=3,  precompute_sim_matrix=True),
    # EnergyKNN(cosine_sim,    class_equality_sim, n_neighbors=10, precompute_sim_matrix=True),
    # EnergyKNN(hamming_sim,   class_equality_sim, n_neighbors=1,  precompute_sim_matrix=True),
    # EnergyKNN(hamming_sim,   class_equality_sim, n_neighbors=3,  precompute_sim_matrix=True),
    # EnergyKNN(hamming_sim,   class_equality_sim, n_neighbors=10, precompute_sim_matrix=True),
]

# %% [markdown]
# Datasets

# %%
import torch
import pickle
sys.path.append(os.path.join(CURRENT_FOLDER, 'benchmark/maintenance/preprocess/'))
import _utils as dataset_utils
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer

# refers to https://cora.ucc.ie/server/api/core/bitstreams/39193798-3fe0-461a-b1b6-3d9cffd108d3/content
BENCHMARK_FOLDER = os.path.join(CURRENT_FOLDER, )
RESULT_FOLDER = os.path.join(BENCHMARK_FOLDER, "results")

APPLY_SCALING = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RECOMPUTE = False

# for the weight estimation experiment
RESULT_FOLDER = os.path.join(BENCHMARK_FOLDER, "results")
N_SPLITS = 2
# baseline_file = os.path.join(BENCHMARK_FOLDER, "baseline.csv")
# baseline_df = pd.read_csv(baseline_file, header=0)

dataset_names = []
datasets = []
paths = []
for dataset in dataset_utils.DATASETS:

    state_dict = dataset_utils.load_dataset_from_pickle(dataset)
    X = state_dict["X"]
    y = state_dict["y"]
    y_values = np.unique(y)
    numeric_columns = state_dict["numeric_columns"]
    symbolic_columns = state_dict["symbolic_columns"]

    if APPLY_SCALING:
        scaler = StandardScaler()
        X = scaler.fit_transform(X, y)

    # 10 splits of 60%, 20%, with 20% test set
    try:
        splits = [{"train_full_index": train, "test_index": test, "fold": i} for i, (train, test) in enumerate(StratifiedShuffleSplit(n_splits=N_SPLITS, random_state=0, test_size=0.2).split(X, y))]
    except ValueError:
        try:
            splits = [{"train_full_index": train, "test_index": test, "fold": i} for i, (train, test) in enumerate(StratifiedShuffleSplit(n_splits=1, random_state=1, test_size=0.2).split(X, y))]
            print("failed to handle", dataset, "used fallback one split")
        except ValueError:
            print("failed to handle", dataset, "skipping")
            continue

    dataset_names.append(dataset.replace("+", " ").title())

    result_folder_dataset = os.path.join(RESULT_FOLDER, dataset)
    paths.append({
        "checkpoint_folder": os.path.join(result_folder_dataset, "checkpoints"),
        "figure": os.path.join(result_folder_dataset, "performance.png"),
        "dataframe": os.path.join(result_folder_dataset, "summary.pkl")
    })

    for dic in splits:
            X_train_full, y_train_full = X[dic["train_full_index"]], y[dic["train_full_index"]]
            X_test, y_test = X[dic["test_index"]], y[dic["test_index"]]
            X_train, X_ref, y_train, y_ref = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=0)
            dic["train_full"] = (X_train_full, y_train_full)
            dic["train"] = (X_train, y_train)
            dic["ref"] = (X_ref, y_ref)
            dic["test"] = (X_test, y_test)

    datasets.append({
        "X": X,
        "y": y,
        "y_values": y_values,
        "splits": splits,
        "numeric_columns": numeric_columns,
        "symbolic_columns": symbolic_columns,
    })

    if APPLY_SCALING:
         datasets[-1]["scaler"] = scaler

# %%
dataset_names

# %% [markdown]
# Train all the models and apply maintenance

# %%
from meatcube2.metrics import confidence, clf_prediction_summary
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
dfs = dict()
print (DEVICE)


#torch.cuda.memory._record_memory_history()

with logging_redirect_tqdm():
    with torch.no_grad():
        for dataset, dataset_name in (pbar1:=tqdm(list(zip(datasets, dataset_names)))):
            pbar1.set_description(dataset_name)
            for split_id, split in (pbar2:=tqdm(list(enumerate(dataset["splits"])), leave=False)):
                #print("#", end="")
                (X_train, y_train) = split["train"]
                (X_ref, y_ref) = split["ref"]
                (X_test, y_test) = split["test"]
                unique, counts = np.unique(y_test, return_counts=True)
                pbar2.set_description(f"Split ( single-class baseline: {(counts / counts.sum())})")
                
                pbar3 = tqdm(list(zip(classifiers, classifier_names, classifier_properties)), leave=False)
                pbar4 = tqdm(range(len(y_train)), leave=False)
                
                model_str = ""
                # define a custom metric to account for the test set performance
                def test_ref_clf_prediction_summary(cb, X_ref, y_ref):
                    #torch.cuda.memory._dump_snapshot()
                    scores = {
                        "ref_"+k: v for k, v in clf_prediction_summary(cb, X_ref, y_ref).items()
                    }
                    scores.update({
                        "test_"+k: v for k, v in clf_prediction_summary(cb, X_test, y_test).items()
                    })
                    pbar4.update()
                    pbar4.set_description(str({"|CB|": len(cb), "ref. acc.": scores['ref_accuracy'], "test acc.": scores['test_accuracy']}))
                    #pbar3.set_postfix({"|CB|": len(cb), "ref. acc.": scores['ref_accuracy'], "test acc.": scores['test_accuracy']})
                    return scores
                
                # apply the model
                for model, model_name, model_pp in pbar3:
                    pbar4.reset()
                    model_str = "Model: "+ model_pp["model"]
                    pbar3.set_description("Model: "+ model_pp["model"])
                    #print("\t", model_pp, dataset_name, len(y_train),  len(y_ref))
                    maintainer = CBClassificationMaintainer(
                        model,
                        memorize_estimators=True,
                        scoring=test_ref_clf_prediction_summary,
                        refit="ref_accuracy",
                        patience=-1)
                    maintainer.fit(X_train, y_train, X_ref, y_ref, fit_kwargs={"device": DEVICE})

                    df_ = pd.DataFrame.from_records(maintainer.results_)
                    df_["parameters"] = [tuple(model_pp.items())]*len(df_)
                    dfs[(model_name, dataset_name, split_id)] = df_
                    #torch.cuda.memory._dump_snapshot()
                pbar4.close()

                df = pd.concat({k: v for k, v in dfs.items() if (k[1] == dataset_name) and (k[2] == split_id)})
                df.to_pickle(f"benchmark_df_{dataset_name}_split_{split_id}.pkl")
            df = pd.concat({k: v for k, v in dfs.items() if (k[1] == dataset_name)})
            df.to_pickle(f"benchmark_df_{dataset_name}.pkl")

    # %%
    index = list(dfs.keys())
    df = pd.concat(dfs)
    df.to_pickle("benchmark_df_all.pkl")


