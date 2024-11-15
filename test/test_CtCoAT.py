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

def test_manual_energy_cb(manual_energy_CtCoAT):
        # energy of the case
    for manual_energy_ in manual_energy_CtCoAT:
        cb = CtCoAT(euclidean_sim, class_equality_sim)
        cb.fit(manual_energy_["CB X"], manual_energy_["CB y"])
        calculated_e = cb.energy_cb()
        e = manual_energy_["energy_cb"]
        assert abs(calculated_e - e) <= manual_energy_["epsilon"], f'{calculated_e} != {e}'        
    
def test_manual_energy_case_new(manual_energy_CtCoAT):
    # energy of the case
    for manual_energy_ in manual_energy_CtCoAT:
        cb = CtCoAT(euclidean_sim, class_equality_sim)
        cb.fit(manual_energy_["CB X"], manual_energy_["CB y"])
        for X_, y_, e in zip(manual_energy_["X"], manual_energy_["y"], manual_energy_["energy_case_new"]):
            calculated_e = cb.energy_case_new(X_, y_)
            assert abs(calculated_e - e) <= manual_energy_["epsilon"], f"error of {calculated_e - e}: |{calculated_e=} - {e=}|"

def test_manual_energy_case_new_through_cb(manual_energy_CtCoAT):
    # energy of the case
    for manual_energy_ in manual_energy_CtCoAT:
        cb = CtCoAT(euclidean_sim, class_equality_sim)
        cb.fit(manual_energy_["CB X"], manual_energy_["CB y"])
        for X_, y_, e in zip(manual_energy_["X"], manual_energy_["y"], manual_energy_["energy_case_new"]):
            calculated_e = cb.add(X_, y_).energy_cb() - cb.energy_cb() 
            assert abs(calculated_e - e) <= manual_energy_["epsilon"], f"error of {calculated_e - e}: |{calculated_e=} - {e=}|"

def test_manual_hinge_cb(manual_energy_CtCoAT):
    # energy of the case
    for manual_energy_ in manual_energy_CtCoAT:
        cb = CtCoAT(euclidean_sim, class_equality_sim)
        cb.fit(manual_energy_["CB X"], manual_energy_["CB y"])
        for hinge_margin, hinge_case_new in manual_energy_["hinge_case_new"].items():
            # check with no aggregation
            calculated_e_ = cb.loss_cb(manual_energy_["X"], manual_energy_["y"], strategy="hinge", margin=hinge_margin, aggregation=None)
            for calculated_e, e in zip(calculated_e_, hinge_case_new):
                assert abs(calculated_e - e) <= manual_energy_["epsilon"], f"error of {calculated_e - e}: |{calculated_e=} - {e=}| (margin={hinge_margin})"

            # check with mean aggregation
            calculated_e = cb.loss_cb(manual_energy_["X"], manual_energy_["y"], strategy="hinge", margin=hinge_margin, aggregation="mean")
            assert abs(calculated_e - np.mean(hinge_case_new)) <= manual_energy_["epsilon"], f"error of {calculated_e - np.mean(hinge_case_new)}: |{calculated_e=} - {np.mean(hinge_case_new)=}| (margin={hinge_margin})"

            # check with sum aggregation
            calculated_e = cb.loss_cb(manual_energy_["X"], manual_energy_["y"], strategy="hinge", margin=hinge_margin, aggregation="sum")
            assert abs(calculated_e - np.sum(hinge_case_new)) <= manual_energy_["epsilon"], f"error of {calculated_e - np.sum(hinge_case_new)}: |{calculated_e=} - {np.sum(hinge_case_new)=}| (margin={hinge_margin})"

            # also check one by one
            for i in range(manual_energy_["X"].shape[0]):
                e = hinge_case_new[i]
                calculated_e = cb.loss_cb(manual_energy_["X"][i:i+1], manual_energy_["y"][i:i+1], strategy="hinge", margin=hinge_margin)
                assert abs(calculated_e - e) <= manual_energy_["epsilon"], f"error of {calculated_e - e}: |{calculated_e=} - {e=}| (margin={hinge_margin})"