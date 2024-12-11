import sys, os
try:
    CURRENT_FOLDER = os.path.dirname(__file__) # normal way
except NameError:
    CURRENT_FOLDER = globals()['_dh'][0] # jupyter notebook way
sys.path.insert(0,os.path.join(CURRENT_FOLDER))

# we load meatcube2 
import benchmark_one
from tqdm import tqdm

# %% [markdown]
# Datasets

# %%
import torch
import pickle
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from tqdm.contrib.logging import logging_redirect_tqdm
from logging import warning

if __name__ == "__main__":
    with logging_redirect_tqdm():
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        warning(f"Using device {DEVICE} when compatible")

        sys_argv = sys.argv

        # for the weight estimation experiment
        N_SPLITS = 2
        # baseline_file = os.path.join(BENCHMARK_FOLDER, "baseline.csv")
        # baseline_df = pd.read_csv(baseline_file, header=0)

        dataset_names = []
        datasets = []
        paths = []

        # just to get the --no_tqdm flag
        sys.argv = sys_argv[:1] + [benchmark_one.DATASETS[0]] + ["-a", list(benchmark_one.CB_LEARNERS.keys())[0]] + ["-c", list(benchmark_one.CLASSIFIERS.keys())[0]] + sys_argv[1:]
        dummy_args = benchmark_one.parse_args()
        
        for dataset in (pbar_datasets:=tqdm(list(benchmark_one.DATASETS), leave=False, disable=dummy_args.no_tqdm)):
            pbar_datasets.set_description(str(dataset))
                    
            for model in (pbar_models:=tqdm(list(benchmark_one.CLASSIFIERS.keys()), leave=False, disable=dummy_args.no_tqdm)):
                pbar_models.set_description(str(model))
                    
                for algo in (pbar_algos:=tqdm(list(benchmark_one.CB_LEARNERS.keys()), leave=False, disable=dummy_args.no_tqdm)):
                    pbar_algos.set_description(str(algo))
                    
                    # artificially insert the dataset, algo, and model to the arguments
                    sys.argv = sys_argv[:1] + [dataset] + ["-a", algo] + ["-c", model] + sys_argv[1:]
                    #print(sys.argv)
                    args = benchmark_one.parse_args()
                    benchmark_one.main(device=DEVICE)
