# %%
import bz2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
from packaging.version import Version
from tqdm import tqdm
from logging import warning, info, error
from tqdm.contrib.logging import logging_redirect_tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from functools import partial

# as the code is loaded from a subfolder, we use the following snippet to add meatcube2 to the import path
# for a normal usage with meatcube2 installed, the two following lines are unnecessary
import sys, os
try:
    CURRENT_FOLDER = os.path.dirname(__file__) # normal way
except NameError:
    CURRENT_FOLDER = globals()['_dh'][0] # jupyter notebook way
sys.path.append(os.path.join(CURRENT_FOLDER, ".."))

# we load meatcube2 
from meatcube2.models import MeATCubeCB, CtCoAT, CtCoATNaive, EnergyKNN, EnergyClf, AbstractEnergyBasedClassifier
from meatcube2.cb_maintenance import EnergyCompress, CNNR, InamoriISelSingleStep
from meatcube2.cb_maintenance.inamori import INAMORY_I_SEL_METHODS
from meatcube2.metrics import confidence, clf_prediction_summary

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

# load the preprocessed dataset
from benchmark.maintenance.preprocess.dataset_utils import load_dataset_from_pickle, DATASETS

VERSION = "0.2"
RANDOM_STATE = 0


# refers to https://cora.ucc.ie/server/api/core/bitstreams/39193798-3fe0-461a-b1b6-3d9cffd108d3/content
BENCHMARK_FOLDER = os.path.join(CURRENT_FOLDER)
RESULT_FOLDER = os.path.join(BENCHMARK_FOLDER, "results", f"v{VERSION}")

APPLY_SCALING = True

CLASSIFIERS = {
    "CoAT":         MeATCubeCB,
    "CtCoAT":       CtCoAT,
    "kNN":          (lambda sim_X, sim_y, args: EnergyKNN(sim_X=sim_X, sim_y=sim_y, n_neighbors=args.k, precompute_sim_matrix=True)),
    "SVM-poly":     (lambda sim_X, sim_y, args: EnergyClf(SVC(kernel="poly", probability=True))),
    "SVM-rbf":     (lambda sim_X, sim_y, args: EnergyClf(SVC(kernel="rbf", probability=True))),
    "SVM-linear":     (lambda sim_X, sim_y, args: EnergyClf(SVC(kernel="linear", probability=True))),
}


CB_LEARNERS = {
    "EnergyCompress":   (lambda args: EnergyCompress),
    "CNNR":             (lambda args: CNNR),
    "CkNNR":            (lambda args: partial(CNNR, n_neighbors=args.k)),

    'cnn':   (lambda args: partial(InamoriISelSingleStep, method='cnn')),
    'enn':   (lambda args: partial(InamoriISelSingleStep, method='enn')),
    'icf':   (lambda args: partial(InamoriISelSingleStep, method='icf')),
    'lssm':  (lambda args: partial(InamoriISelSingleStep, method='lssm')),
    'ldis':  (lambda args: partial(InamoriISelSingleStep, method='ldis')),
    'cdis':  (lambda args: partial(InamoriISelSingleStep, method='cdis')),
    'xldis': (lambda args: partial(InamoriISelSingleStep, method='xldis')),
    'psdsp': (lambda args: partial(InamoriISelSingleStep, method='psdsp')),
    'ib3':   (lambda args: partial(InamoriISelSingleStep, method='ib3')),
    'egdis': (lambda args: partial(InamoriISelSingleStep, method='egdis')),
    #'cis':   (lambda args: partial(InamoriISelSingleStep, method='cis')),
}

# argument parsing
import argparse
def parse_args(arg_string=None)  -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    #parser.add_argument("method", type=str, choices=["learn", "fit", "draw", "plot"])
    
    parser.add_argument("dataset", type=str, help="the name of the dataset",  choices=DATASETS)

    parser.add_argument(
        "-a", "--algo", type=str, help="the case base learning algorithm", choices=CB_LEARNERS.keys(), default="EnergyCompress"
    )
    parser.add_argument(
        "-c", "--classifier", type=str, help="the case base prediction algorithm", choices=CLASSIFIERS.keys()
    )
    parser.add_argument(
        "-m", "--margin", type=float, default=1e-1,
        help="the margin to use in the hinge loss of the energy-compress algorithm (default: 1e-1)"
    )
    parser.add_argument(
        "-k", "--k", type=int, default=7,
        help="the k parameter for kNN, CNNR, or APC algorithms (default: 7)"
    )
    parser.add_argument(
        "-f", "--folds", type=int, default=10,
        help="the number of folds used for cross validation (default: 10)"
    )

    parser.add_argument(
        "-T", "--size_T", type=float, default=.2,
        help="the size of the test set, as an integer or as a proportion of the full data, or 0 to use all the remaining data(default:.2)"
    )
    parser.add_argument(
        "-t", "--max_size_T", type=int, default=100,
        help="max the size of the test set, or 0 for no limit (default:100)"
    )
    parser.add_argument(
        "-V", "--size_V", type=float, default=.2,
        help="the size of the validation, as an integer or as a proportion of the full data set (default:.2)"
    )
    parser.add_argument(
        "-v", "--max_size_V", type=int, default=100,
        help="max size of the validation set, used to run some of the CB learning algos, or 0 for no limit (default:100)"
    )
    parser.add_argument(
        "-S", "--size_S", type=float, default=0,
        help="the size of the initial case base, as an integer or as a proportion of the full data set, or 0 to use all the remaining data (default:0)",
    )
    parser.add_argument(
        "-s", "--max_size_S", type=float, default=50,
        help="the size of the initial case base, or 0 for no limit (default:50)",
    )
    parser.add_argument(
        "--no_tqdm", action='store_true',
        help="Use this flag to disable progress bars",
    )
    parser.add_argument(
        "--force_recompute", action='store_true',
        help="Use this flag to force running the experiment even if a corresponding file already exists",
    )


    args = parser.parse_args(arg_string)

    # handle size matters
    args.dataset_code = args.dataset
    args.dataset_name = args.dataset_code.replace("+", " ").title()
    args.dataset = load_dataset_from_pickle(args.dataset_code)


    dataset_len = len(args.dataset["y"])
    if 0 < args.size_T < 1: args.size_T = (dataset_len*args.size_T)//1
    elif args.size_T >= 1: args.size_T = min(dataset_len, args.size_T//1)
    else: args.size_T = 0
    if 0 < args.size_V < 1: args.size_V = (dataset_len*args.size_V)//1
    elif args.size_V >= 1: args.size_V = min(dataset_len, args.size_V//1)
    if 0 < args.size_S < 1: args.size_S = (dataset_len*1)//1
    elif args.size_S >= 1: args.size_S = min(dataset_len, args.size_S//1)
    else: args.size_S = 0
    
    # apply max sizes
    if args.max_size_T > 0: args.size_T = min(args.max_size_T, args.size_T)
    if args.max_size_V > 0: args.size_V = min(args.max_size_V, args.size_V)
    if args.max_size_S > 0: args.size_S = min(args.max_size_S, args.size_S)

    # try to combine everything, giving priority to ref, then train if non-zero, then test
    if args.size_S + args.size_V + args.size_T > dataset_len: raise ValueError(f"Not enough data to have {args.size_S + args.size_V + args.size_T=}, got only {dataset_len} samples.")
    if args.size_S == 0 and args.size_T == 0: raise ValueError(f"Cannot have args.size_S == 0 and args.size_T == 0.")
    if args.size_S == 0: args.size_S = dataset_len - args.size_V - args.size_T
    elif args.size_T == 0: args.size_S = dataset_len - args.size_V - args.size_S

    # reapply max sizes
    if args.max_size_T > 0: args.size_T = min(args.max_size_T, args.size_T)
    if args.max_size_V > 0: args.size_V = min(args.max_size_V, args.size_V)
    if args.max_size_S > 0: args.size_S = min(args.max_size_S, args.size_S)

    args.size_T = int(args.size_T)
    args.size_V = int(args.size_V)
    args.size_S = int(args.size_S)

    info(f"{args.size_T=} {args.size_V=} {args.size_S=}")

    # summarize model properties
    args.model_pp = {"model": args.classifier, "compression": args.algo, "$\sigma_X$": "Euclidean", "$\sigma_y$": "class", "k": args.k, "margin": args.margin}

    return args

def make_result_path(args):
    template = f"{args.dataset_code}_{args.algo}_{args.classifier}_k{args.k}_f{args.folds}_m{args.margin}-S{args.size_S}_V{args.size_V}_T{args.size_T}.pkl.bz2"
    return os.path.join(RESULT_FOLDER, template)

def check_if_needs_rerun(args, random_state):
    result_path = make_result_path(args)
    # rerun if the file does not exist
    if not os.path.exists(result_path) or not os.path.isfile(result_path):
        return True
    
    # rerun if the file is too old
    state_dict = load_results(path=result_path)
    if Version(state_dict["version"]) < Version(VERSION): 
        warning(f"results are from an older version ({state_dict['version']} < {VERSION}), rerunning the experiment")
        return True
    elif random_state != state_dict["random_state"]:
        warning(f"results use a different random state ({RANDOM_STATE} != {state_dict['random_state']}), rerunning the experiment")
        return True
    
    return False

def prep_dataset(args, random_state=RANDOM_STATE):
    X = args.dataset["X"]
    y = args.dataset["y"]
    y_values = np.unique(y)
    numeric_columns = args.dataset["numeric_columns"]
    symbolic_columns = args.dataset["symbolic_columns"]

    if APPLY_SCALING:
        scaler = StandardScaler()
        X = scaler.fit_transform(X, y)

    # perform S ∪ V | T split
    try:
        splits = [{"train_full_index": train, "test_index": test, "fold": i} for i, (train, test) in enumerate(StratifiedShuffleSplit(n_splits=args.folds, random_state=random_state, test_size=args.size_V).split(X, y))]
    except ValueError:
        try:
            splits = [{"train_full_index": train, "test_index": test, "fold": i} for i, (train, test) in enumerate(StratifiedShuffleSplit(n_splits=1, random_state=random_state, test_size=args.size_V).split(X, y))]
            warning(f"failed to handle {args.dataset_name} stratified splitting for {args.size_V=} and {args.folds=}, used fallback one fold sucessfully")
        except ValueError as v:
            warning(f"failed to handle {args.dataset_name} stratified splitting for {args.size_V=}, aborting")
            raise v

    # perform S | V split for each fold
    for dic in splits:
        X_train_full, y_train_full = X[dic["train_full_index"]], y[dic["train_full_index"]]
        X_test, y_test = X[dic["test_index"]], y[dic["test_index"]]
        X_train, X_ref, y_train, y_ref = train_test_split(X_train_full, y_train_full, test_size=args.size_V, train_size=args.size_S, random_state=random_state)
        dic["train_full"] = (X_train_full, y_train_full)
        dic["train"] = (X_train, y_train)
        dic["ref"] = (X_ref, y_ref)
        dic["test"] = (X_test, y_test)
        if APPLY_SCALING:
            dic["scaler"] = scaler
    
    return splits, y_values

ARGS_TO_MEMORIZE = ["algo", "classifier", "margin", "k", "folds", "size_T", "size_V", "size_S"]
def save_results(args: argparse.Namespace, records, models, path):
    state_dict = dict()
    state_dict["version"] = VERSION
    state_dict["random_state"] = RANDOM_STATE
    state_dict["args"] = {
        attr: vars(args).get(attr) for attr in ARGS_TO_MEMORIZE
    }
    state_dict["args"]["dataset"] = args.dataset_code
    state_dict["records"] = records
    state_dict["models_cb"] = models

    with bz2.open(path, "wb") as f:
        state_dict = pickle.dump(state_dict, f)
    
def load_results(path=None, args: argparse.Namespace=None):
    if args is not None:
        path = make_result_path(args)
    with bz2.open(path, "rb") as f:
        state_dict = pickle.load(f)
    return state_dict

# %%
import torch
import pickle
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer

def main(args=None, arg_string=None, device='cpu'):
    with logging_redirect_tqdm():
        if args is None:
            args = parse_args(arg_string=arg_string)
        save_file = make_result_path(args)
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        
        if args.force_recompute:
            warning("Forcing rerun, ignoring file existence check for the result file.")
        elif not check_if_needs_rerun(args, RANDOM_STATE):
            warning(f"No rerun required, use --force_recompute to force rerun or delete '{save_file}'")
            return 1
        
        try:
            splits, y_values = prep_dataset(args)
        except ValueError as e:
            error(e)
            return 2

        records = []
        models = dict()
        with torch.no_grad():
            for split_id, split in (pbar_splits:=tqdm(list(enumerate(splits)), leave=False, disable=args.no_tqdm)):
                (X_train, y_train) = split["train"]
                (X_ref, y_ref) = split["ref"]
                (X_test, y_test) = split["test"]
                unique, counts = np.unique(y_test, return_counts=True)
                pbar_splits.set_description(f"Split (single-class baselines accuracy: {(counts / counts.sum())})")
                
                pbar_steps = tqdm(range(len(y_train)), leave=False, disable=args.no_tqdm)
                
                # define a custom metric to account for the test set performance
                def test_ref_clf_prediction_summary(cb, X_ref, y_ref):
                    #torch.cuda.memory._dump_snapshot()
                    scores = {
                        "ref_"+k: v for k, v in clf_prediction_summary(cb, X_ref, y_ref).items()
                    }
                    scores.update({
                        "test_"+k: v for k, v in clf_prediction_summary(cb, X_test, y_test).items()
                    })
                    scores.update(args.model_pp)
                    scores.update({
                        "fold": split_id
                    })
                    pbar_steps.update()
                    pbar_steps.set_description(str({"|CB|": len(cb), "ref. acc.": scores['ref_accuracy'], "test acc.": scores['test_accuracy']}))
                    #pbar3.set_postfix({"|CB|": len(cb), "ref. acc.": scores['ref_accuracy'], "test acc.": scores['test_accuracy']})
                    return scores
                

                pbar_steps.reset()

                # create the model
                if args.classifier.lower().strip() == "MeATCubeCB".lower() or args.classifier == "CoAT".lower():
                    model = MeATCubeCB(euclidean_sim, class_equality_sim, precompute_sim_matrix=True)
                elif args.classifier.lower().strip() == "CtCoAT":
                    model = CtCoAT(euclidean_sim, class_equality_sim, precompute_sim_matrix=True)
                elif args.classifier.lower().strip() == "kNN".lower() or args.classifier.lower().strip() == "EnergyKNN".lower():
                    model = EnergyKNN(euclidean_sim, class_equality_sim, n_neighbors=args.k, precompute_sim_matrix=True)
                elif args.classifier in CLASSIFIERS.keys():
                    model = CLASSIFIERS[args.classifier](euclidean_sim, class_equality_sim, args)
                else: raise ValueError("Unsupported classifier")

                # create the compression algo
                if args.algo in CB_LEARNERS.keys():
                    maintainer = CB_LEARNERS[args.algo](args)(
                        model,
                        memorize_estimators=True,
                        scoring=test_ref_clf_prediction_summary,
                        refit="ref_accuracy",
                        patience=-1)
                else: raise ValueError(f"Unsupported algo: '{args.algo}'")

                # run the compression algo
                maintainer.fit(X_train, y_train, X_ref, y_ref, fit_kwargs={"device": device, "margin": args.margin})

                # )
                records += maintainer.results_
                models[f"fold {split_id}"] = [(model._X, model._y) for model in maintainer.estimators_]
                pbar_steps.close()
    save_results(args, records, models, save_file)
    return 0

if __name__ == "__main__":
    with logging_redirect_tqdm():
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        warning(f"Using device {DEVICE} when compatible")
        status = main(device=DEVICE)
        sys.exit(status)
    
