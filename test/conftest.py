import pytest
from typing import Tuple, List
import pandas as pd
import numpy as np
import numpy.linalg as npl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load meatcube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import meatcube2.models.meatcube_torch as mc



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

def iris_split(X, y, y_values) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # stratified splitting of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

# def iris_cb(X_train:pd.DataFrame, y_train:pd.DataFrame, y_values) -> mc.MeATCubeCB:
#     # create the CB
#     source_similarity = _source_similarity
#     outcome_similarity = _outcome_similarity
#     cb = mc.MeATCubeCB(source_similarity, outcome_similarity)
#     cb.fit(X_train.to_numpy(), y_train.to_numpy())#, y_values, )
#     return cb


# @pytest.fixture()
# def test_cbs(iris_num_cb, iris_str_cb) -> List[mc.CB]:
#     return [iris_num_cb, iris_str_cb]




@pytest.fixture(scope="module")
def manual_energy():
    X_cb = np.array([[-1,0],[1,0]])
    y_cb = np.array([1,0])

    # with |CB|=2, 
    # energy increase for CtCoat should be 
    #   7.5 + 15/2 + 3*simS(X,\bluecircle) - 2*simS(X,\redsquare) + simS(\bluecircle,\redsquare)
    #   7.5 + 15/2 + 3*simS(X,\bluecircle) - 2*simS(X,\redsquare) + simS(\bluecircle,\redsquare)
    manual_energy_ = dict()

    manual_energy_["CB X"] = X_cb
    manual_energy_["CB y"] = y_cb

    N = 10
    np.random.seed(42)
    X = np.random.rand(N,2)
    y = np.random.randint(2,size=N)
    manual_energy_["X"] = X
    manual_energy_["y"] = y

    # energy kNN
    manual_energy_["EnergyKNN"] = dict()
    EnergyKNN_energy_1nn = lambda x, y: 1-1*(
        int(y == 1) if npl.norm(x - X_cb[0]) <= npl.norm(x - X_cb[1]) else int(y == 0)
    )
    EnergyKNN_energy_2nn = lambda x, y: 1-(1/2)*(
        int(y == 1) + int(y == 0)
    )
    """
    kNN(s_t) = argmax_{r ∈ R}(∑_{(s_i,r_i) ∈ CB} simS(s_i,s_t) ⋅ simR(r_i,r) )
    EkNN(s_t, r) = 1 - (1/k) * ∑ {(s_i,r_i) ∈ CB} 1_{N_k}(s_i,s_t) ⋅ simR(r_i,r)

    for CB: 
    {
       ([-1,0], 1),
       ([1,0],  0)
    }
    energy of ([-1,0], 1) is: 
    EkNN([-1,0], 1) = 1 - 1/k * ∑[
        1_{N_k}([-1,0], [-1,0]) * simR(1, 1)
        1_{N_k}([1,0], [-1,0]) * simR(0, 1)
    ]
    for k=1:
    EkNN([-1,0], 1) = 1 - 1/1 * 1 = 0
    EkNN([1,0],  0) = 1 - 1/1 * 1 = 0
    EkNN([-1,0], 0) = 1 - 1/1 * 0 = 1
    EkNN([1,0],  1) = 1 - 1/1 * 0 = 1
    for k=2:
    EkNN([-1,0], 1) = 1 - 1/2 * 1 = 1/2
    EkNN([1,0],  0) = 1 - 1/2 * 1 = 1/2
    EkNN([-1,0], 0) = 1 - 1/2 * 1 = 1/2
    EkNN([1,0],  1) = 1 - 1/2 * 1 = 1/2

    energy of the CB is the sum (or average?) of energies it gives to its own cases (without them in the CB)
    k=1: EkNN() = 1
    k=2: EkNN() = 1

    """
    manual_energy_["EnergyKNN"][1] = {
        "energy_cb": 1,
        "epsilon": 1e-6,
        "energy_case_new": [
            EnergyKNN_energy_1nn(X_, y_)
            for X_, y_ in zip(X, y)
        ],
        "energy_case_from_cb": [
            1, # EkNN([-1,0], 1) for a CB with only ([1,0],  0) inside
            1, # EkNN([1,0],  0) for a CB with only ([-1,0], 1) inside
        ]
    }


    # # CtCoAT
    # expected_energies = np.array(
    #     [7.5 + 3*(euclidean_sim(x,X_cb[1])
    #                if y==1 
    #                else euclidean_sim(x,X_cb[0])) - 2*(euclidean_sim(x,X_cb[0]) if y==1 else euclidean_sim(x,X_cb[1])) + euclidean_sim(X_cb[0],X_cb[1]) for x,y in zip(X,y)])/27
    # manual_energy_["CtCoAT"] = {
        
    # }

    return [manual_energy_]