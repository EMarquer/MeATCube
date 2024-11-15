import pytest
from typing import Tuple, List
import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

# load meatcube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from meatcube2.models import EnergyKNN
from meatcube2.models.energy_knn_backend import KNNEnergyComputations


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



def test_manual_energy_cb(manual_energy_EnergyKNN):
        # energy of the case
    for manual_energy_ in manual_energy_EnergyKNN:
        ks = manual_energy_["k"].keys()
        for k in ks:
            cb = EnergyKNN(euclidean_sim, class_equality_sim, k)
            cb.fit(manual_energy_["CB X"], manual_energy_["CB y"])
            calculated_e = cb.energy_cb()
            e = manual_energy_["k"][k]["energy_cb"]
            assert abs(calculated_e - e) <= manual_energy_["k"][k]["epsilon"], f'{calculated_e} != {e}'        
    
def test_manual_energy_case_new(manual_energy_EnergyKNN):
        # energy of the case
    for manual_energy_ in manual_energy_EnergyKNN:
        ks = manual_energy_["k"].keys()
        for k in ks:
            cb = EnergyKNN(euclidean_sim, class_equality_sim, k)
            cb.fit(manual_energy_["CB X"], manual_energy_["CB y"])
            for X_, y_, e in zip(manual_energy_["X"], manual_energy_["y"], manual_energy_["k"][k]["energy_case_new"]):
                calculated_e = cb.energy_case_new(X_, y_)
                assert abs(calculated_e - e) <= manual_energy_["k"][k]["epsilon"], f"error of {calculated_e - e}: |{calculated_e=} - {e=}|"
        
def test_manual_1nn_mask(manual_energy_EnergyKNN):
    """Check that for the case in the CB, KNNEnergyComputations.knn_mask for k=1 produces a mask with True on the diagonal.
    
    This corresponds to checking that each case is its nearest neighbor, which should be true.
    """
    for manual_energy_ in manual_energy_EnergyKNN:
        cb = EnergyKNN(euclidean_sim, class_equality_sim, 1, precompute_sim_matrix=True)
        cb.fit(manual_energy_["CB X"], manual_energy_["CB y"])
        mask = KNNEnergyComputations.knn_mask(cb.X_sim_matrix_, k=1)
        assert (mask.cpu() == torch.eye(len(cb), dtype=bool)).all(), f"{mask.cpu()} != {torch.eye(len(cb), dtype=bool)}"
           
def test_manual_maxnn_mask(manual_energy_EnergyKNN):
    """Check if for any input X, KNNEnergyComputations.knn_mask produces a mask full of True when k is the size of the CB"""
    for manual_energy_ in manual_energy_EnergyKNN:
        cb = EnergyKNN(euclidean_sim, class_equality_sim, len(manual_energy_["CB y"]), precompute_sim_matrix=True)
        cb.fit(manual_energy_["CB X"], manual_energy_["CB y"])
        
        # similarity between every case in CB and cases in X
        sim_S = cb._source_sim_vect(torch.tensor(manual_energy_["X"])).transpose(-1,-2)
        
        mask = KNNEnergyComputations.knn_mask(sim_S, k=len(cb))
        assert (
            (mask.cpu() == torch.ones((len(cb), len(manual_energy_["X"])), dtype=bool)).all(),
            f"{mask.cpu()} != { torch.ones((len(cb), len(manual_energy_['X'])), dtype=bool)}"
        )
        

@pytest.mark.parametrize('k', [1,2,3,5,10])
@pytest.mark.parametrize('n_cb', [10,20,30,50,100])
@pytest.mark.parametrize('n_classes', [2,3,5,10])
def test_knn_sklearn_same_predictions(k, n_cb, n_classes, n_test=100, n_dims=5, random_seed = 42):
    np.random.seed(random_seed)

    X_cb = np.random.rand(n_cb,n_dims)
    y_cb = np.random.randint(n_classes,size=n_cb)
    X = np.random.rand(n_test,n_dims)
    #y = np.random.randint(n_classes,size=n_test)

    cb = EnergyKNN(euclidean_sim, class_equality_sim, n_neighbors=k, precompute_sim_matrix=True)
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_cb, y_cb)
    cb.fit(X_cb, y_cb)

    predictions_knn = knn.predict(X)
    predictions_cb = cb.predict(X)

    assert (predictions_knn == predictions_cb).all(), f'{predictions_knn=}!={predictions_cb=}'