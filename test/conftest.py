import pytest
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load meatcube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import meatcube as mc

@pytest.fixture(scope="module")
def iris():
    iris = load_iris(as_frame=True)
    return iris

@pytest.fixture(scope="module")
def iris_string(iris) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
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
def iris_string_split(iris_string) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return iris_split(*iris_string)

@pytest.fixture(scope="module")
def iris_num_split(iris_num) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return iris_split(iris_num)

@pytest.fixture()
def iris_string_cb(iris_string, iris_string_split) -> mc.CB:
    X, y, y_values = iris_string
    X_train, X_test, y_train, y_test = iris_string_split
    return iris_cb(X_train, y_train, y_values)

@pytest.fixture()
def iris_num_cb(iris_num, iris_num_split) -> mc.CB:
    X, y, y_values = iris_num
    X_train, X_test, y_train, y_test = iris_num_split
    return iris_cb(X_train, y_train, y_values)

def iris_split(X, y, y_values) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # stratified splitting of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def iris_cb(X_train, y_train, y_values) -> mc.CB:
    # create the CB
    source_similarity = lambda x,y: np.exp(- np.linalg.norm(x - y))
    outcome_similarity = lambda x,y: (True if x == y else False)
    cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)
    return cb
