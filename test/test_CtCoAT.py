import pytest
from typing import Tuple, List
import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

# load meatcube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from meatcube2.models import CtCoAT


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

def test_manual_energy_cb(manual_energy):
        # energy of the case
    for manual_energy_ in manual_energy:
        cb = CtCoAT(euclidean_sim, class_equality_sim)
        cb.fit(manual_energy_["CB X"], manual_energy_["CB y"])
        calculated_e = cb.energy_cb()
        e = manual_energy_["CtCoAT"]["energy_cb"]
        assert abs(calculated_e - e) <= manual_energy_["CtCoAT"]["epsilon"], f'{calculated_e} != {e}'        
    
def test_manual_energy_case_new(manual_energy):
        # energy of the case
    for manual_energy_ in manual_energy:
        cb = CtCoAT(euclidean_sim, class_equality_sim)
        cb.fit(manual_energy_["CB X"], manual_energy_["CB y"])
        for X_, y_, e in zip(manual_energy_["X"], manual_energy_["y"], manual_energy_["CtCoAT"]["energy_case_new"]):
            calculated_e = cb.energy_case_new(X_, y_)
            assert abs(calculated_e - e) <= manual_energy_["CtCoAT"]["epsilon"], f"error of {calculated_e - e}: |{calculated_e=} - {e=}|"

def test_manual_energy_case_new_through_cb(manual_energy):
        # energy of the case
    for manual_energy_ in manual_energy:
        cb = CtCoAT(euclidean_sim, class_equality_sim)
        cb.fit(manual_energy_["CB X"], manual_energy_["CB y"])
        for X_, y_, e in zip(manual_energy_["X"], manual_energy_["y"], manual_energy_["CtCoAT"]["energy_case_new"]):
            calculated_e = cb.add(X_, y_).energy_cb() - cb.energy_cb() 
            assert abs(calculated_e - e) <= manual_energy_["CtCoAT"]["epsilon"], f"error of {calculated_e - e}: |{calculated_e=} - {e=}|"
