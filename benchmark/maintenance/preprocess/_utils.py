import pandas as pd
import os, sys
import pickle
import numpy as np
from typing import List

"""
To create all the preprocessing files:
echo datasets/*/ | tr + _ | tr ' ' '\n' | sed -r 's/datasets\/(.+?)\//preprocess\/\1.py/g' | xargs touch
"""

THIS_FOLDER = os.path.dirname(__file__)
def auto_dataset_name_from_preprocess_file(_file_):
    dataset_name = os.path.basename(_file_)
    dataset_name = dataset_name.replace(".py", "")
    dataset_name = dataset_name.replace("_", "+")
    return dataset_name
def auto_dataset_path_from_preprocess_file(_file_):
    dataset_name = auto_dataset_name_from_preprocess_file(_file_)
    return os.path.abspath(os.path.join(THIS_FOLDER, "..", "datasets", dataset_name))
def auto_data_path_from_preprocess_file(_file_, extension='.data'):
    dataset_folder = auto_dataset_path_from_preprocess_file(_file_)
    data_files = [
        os.path.abspath(os.path.join(dataset_folder, file_name)) for file_name in os.listdir(dataset_folder) if file_name.endswith(extension)
    ]
    return data_files
def auto_pickle_path_from_dataset_name(dataset_name):
    return os.path.abspath(os.path.join(THIS_FOLDER, "..", "preprocessed", f"{dataset_name}.pkl"))
def auto_pickle_path_from_preprocess_file(_file_):
    dataset_name = auto_dataset_name_from_preprocess_file(_file_)
    return auto_pickle_path_from_dataset_name(dataset_name)

def enforce_num(df, numeric_columns):
    for num_col in numeric_columns:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

def save_dataset_as_pickle(np_X: np.ndarray, np_y: np.ndarray, numeric_columns: List[int], _file_, y_meaning=None):
    state_dict = {
        "X": np_X,
        "y": np_y,
        "y_meaning": y_meaning,
        "numeric_columns": numeric_columns,
        "symbolic_columns": sorted(set(range(np_X.shape[1])).difference(numeric_columns))
    }
    with open(auto_pickle_path_from_preprocess_file(_file_), "wb") as f:
        pickle.dump(state_dict, f)
def load_dataset_from_pickle(dataset_name):
    """
    :returns: dict with the following keys
        "X": np_X,
        "y": np_y,
        "y_meaning": list of values associated with class indices
        "numeric_columns": numeric_columns,
        "symbolic_columns": list symbolic_columns
    """
    with open(auto_pickle_path_from_dataset_name(dataset_name), "rb") as f:
        state_dict = pickle.load(f)
    return state_dict

def symbolic_to_symbol_ids(series: pd.Series):
    values = sorted(series.unique().tolist())
    return series.apply(lambda v: values.index(v)), values
def pd_to_numpy(X: pd.DataFrame, y: pd.Series, numeric_columns):
    np_X, np_y = X.to_numpy(), y.to_numpy()
    column_idx = X.columns.tolist().index
    numeric_columns_idx = [column_idx(c) for c in numeric_columns]
    return np_X, np_y, numeric_columns_idx
def drop_missing(df: pd.DataFrame, missing_values=["?"]):
    missing_mask = ~(
            sum([(df[col].apply(lambda x: x in missing_values)) for col in df.columns]) # if any attribute is "?", remove the row
    ).astype(bool)
    return df[missing_mask]
# # preprocess
# df = df.dropna(axis=0)