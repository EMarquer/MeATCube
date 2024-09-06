import pytest
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load meatcube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import meatcube2.meatcube_torch as mc



def _source_similarity(x,y): return np.exp(- np.linalg.norm(x - y))
def _outcome_similarity(x,y): return  (True if x == y else False)


@pytest.fixture(scope="module")
def iris():
    iris = load_iris(as_frame=True)
    return iris

@pytest.fixture(scope="module")
def iris_str(iris) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    iris = load_iris(as_frame=True)

    X: pd.DataFrame = iris["data"] # source
    y = iris["target"] # target

    # to test with strings as labels
    y = y.apply(lambda x: iris["target_names"][x]) 
    y_values = iris["target_names"]

    return X, y, y_values

@pytest.fixture(scope="module")
def iris_num(iris) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    X: pd.DataFrame = iris["data"] # source
    y = iris["target"] # target
    y_values = np.unique(y)
    return X, y, y_values

@pytest.fixture(scope="module")
def iris_str_split(iris_str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return iris_split(*iris_str)

@pytest.fixture(scope="module")
def iris_num_split(iris_num) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return iris_split(*iris_num)

@pytest.fixture(scope="module")
def iris_cb_train_size(iris_num_split) -> int:
    return iris_num_split[0].shape[0]

@pytest.fixture(scope="module") # /!\ do not do modifications on the cbs
def iris_str_cb(iris_str, iris_str_split) -> mc.MeATCubeCB:
    X, y, y_values = iris_str
    X_train, X_test, y_train, y_test = iris_str_split
    return iris_cb(X_train, y_train, y_values)

@pytest.fixture(scope="module") # /!\ do not do modifications on the cbs
def iris_num_cb(iris_num, iris_num_split) -> mc.MeATCubeCB:
    X, y, y_values = iris_num
    X_train, X_test, y_train, y_test = iris_num_split
    return iris_cb(X_train, y_train, y_values)

def iris_split(X, y, y_values) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # stratified splitting of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def iris_cb(X_train:pd.DataFrame, y_train:pd.DataFrame, y_values) -> mc.MeATCubeCB:
    # create the CB
    source_similarity = _source_similarity
    outcome_similarity = _outcome_similarity
    cb = mc.MeATCubeCB(source_similarity, outcome_similarity)
    cb.fit(X_train.to_numpy(), y_train.to_numpy())#, y_values, )
    return cb


# @pytest.fixture()
# def test_cbs(iris_num_cb, iris_str_cb) -> List[mc.CB]:
#     return [iris_num_cb, iris_str_cb]
