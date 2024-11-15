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
def manual_cb():
    X_cb = np.array([[-1,0],[1,0]])
    y_cb = np.array([1,0])

    manual_energy_ = dict()

    manual_energy_["CB X"] = X_cb
    manual_energy_["CB y"] = y_cb

    N = 10
    np.random.seed(42)
    X = np.random.rand(N,2)
    y = np.random.randint(2,size=N)
    manual_energy_["X"] = X
    manual_energy_["y"] = y

    return [manual_energy_]

@pytest.fixture(scope="module")
def manual_energy_EnergyKNN(manual_cb):
    manual_energy_ = {**manual_cb[0]}

    X_cb = manual_energy_["CB X"]
    y_cb = manual_energy_["CB y"]
    X = manual_energy_["X"]
    y = manual_energy_["y"]

    # energy kNN
    manual_energy_["k"] = dict()
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
    manual_energy_["k"][1] = {
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
    manual_energy_["k"][2] = {
        "energy_cb": 1,
        "epsilon": 1e-6,
        "energy_case_new": [
            EnergyKNN_energy_2nn(X_, y_)
            for X_, y_ in zip(X, y)
        ],
        "energy_case_from_cb": [
            1, # EkNN([-1,0], 1) for a CB with only ([1,0],  0) inside
            1, # EkNN([1,0],  0) for a CB with only ([-1,0], 1) inside
        ]
    }
    return [manual_energy_]

@pytest.fixture(scope="module")
def manual_energy_CtCoAT(manual_cb):
    manual_energy_ = {**manual_cb[0]}

    X_cb = manual_energy_["CB X"]
    y_cb = manual_energy_["CB y"]
    X = manual_energy_["X"]
    y = manual_energy_["y"]

    # CtCoAT
    # with |CB|=2, 
    # energy increase for CtCoat should be 
    #   7.5 + 15/2 + 3*simS(X,\bluecircle) - 2*simS(X,\redsquare) + simS(\bluecircle,\redsquare)
    """
    E CoAT(x,y)=Γ(σS,σR,CB∪{(x,y)}) - Γ(σS,σR,CB)
    for 
    Γ(σS,σR,CB)= ∑_(i,j,k) (1 - [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)])/2
               =  1/2*(∑_(i,j,k) (1 - [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)]))
               =  1/2*(∑_(i,j,k) 1 - ∑_(i,j,k) ([σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)]))
               = (|CB|**3)/2 - 1/2*∑_(i,j,k) ([σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)]))

    so for a |CB| of 2
    
    Γ(σS,σR,CB) = 8/2 - 1/2*(
      [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 0, j = 0, k = 0
      [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 0, j = 0, k = 1
      [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 0, j = 1, k = 0
      [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 0, j = 1, k = 1
      [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 1, j = 0, k = 0
      [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 1, j = 0, k = 1
      [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 1, j = 1, k = 0
      [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)]           i = 1, j = 1, k = 1
    )
                = 8/2 - 1/2*(
      [σS(s0,s0) - σS(s0,s1)]*[σR(r0,r0)-σR(r0,r1)] +         i = 0, j = 0, k = 1
      [σS(s0,s1) - σS(s0,s0)]*[σR(r0,r1)-σR(r0,r0)] +         i = 0, j = 1, k = 0
      [σS(s1,s0) - σS(s1,s1)]*[σR(r1,r0)-σR(r1,r1)] +         i = 1, j = 0, k = 1
      [σS(s1,s1) - σS(s1,s0)]*[σR(r1,r1)-σR(r1,r0)] +         i = 1, j = 1, k = 0
      0 +         i = 0, j = 0, k = 0
      0 +         i = 0, j = 1, k = 1
      0 +         i = 1, j = 0, k = 0
      0           i = 1, j = 1, k = 1
    )
                = 8/2 - 1/2*(
      [1 - σS(s0,s1)]*[1-σR(r0,r1)] +         i = 0, j = 0, k = 1
      [σS(s0,s1) - 1]*[σR(r0,r1)-1] +         i = 0, j = 1, k = 0
      [σS(s1,s0) - 1]*[σR(r1,r0)-1] +         i = 1, j = 0, k = 1
      [1 - σS(s1,s0)]*[1-σR(r1,r0)]           i = 1, j = 1, k = 0
    )
                = 8/2 - 1/2*(
      [1 - σS(s0,s1)]*[1-σR(r0,r1)] +         i = 0, j = 0, k = 1
      -[1-σS(s0,s1)]*-[1-σR(r0,r1)] +         i = 0, j = 1, k = 0
      -[1-σS(s1,s0)]*-[1-σR(r1,r0)] +         i = 1, j = 0, k = 1
      [1 - σS(s1,s0)]*[1-σR(r1,r0)]           i = 1, j = 1, k = 0
    )
                = 8/2 - 1/2*4*([1 - σS(s0,s1)]*[1-σR(r0,r1)])
                                              as σR(r1,r0) = σR(r0,r1) and similarly for σS
    )
    = 4 - 2*(
        1 +
        -σR(r0,r1) +
        -σS(s0,s1) +
        σS(s0,s1)*σR(r0,r1)
    )
    Γ(σS,σR,CB) = 4 - 2 + 2*σR(r0,r1) + 2*σS(s0,s1) - 2*σS(s0,s1)*σR(r0,r1)
    for |CB|={(s0,r0), (s1,r1)}

    in particular for our case using class equality for σR:
    Γ(σS,σR,CB) = 2 + 2*0 + 2*σS(s0,s1) - 2*σS(s0,s1)*0 = 2 + 2*σS(s0,s1)
    σS([1,0],[-1,0]) = e(-2) for Euclidean similarity
    and the energy of the CB is Γ(σS,σR,CB) = 2 + 2*e(-2)
    )


    Now for Γ(σS,σR,CB∪{(st,rt)}):
    Γ(σS,σR,CB∪{(st,rt)}) = (|CB|**3)/2 - 1/2*∑_(i,j,k) ([σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)]))
    = 27/2 - 1/2*(       skipping any j=k that will result in 0
        [σS(s0,s0) - σS(s0,s1)]*[σR(r0,r0)-σR(r0,r1)] +                         i = 0, j = 0, k = 1
        [σS(s0,s1) - σS(s0,s0)]*[σR(r0,r1)-σR(r0,r0)] +                         i = 0, j = 1, k = 0
        [σS(s0,s0) - σS(s0,st)]*[σR(r0,r0)-σR(r0,rt)] +                         i = 0, j = 0, k = t
        [σS(s0,st) - σS(s0,s0)]*[σR(r0,rt)-σR(r0,r0)] +                         i = 0, j = t, k = 0
        [σS(s0,s1) - σS(s0,st)]*[σR(r0,r1)-σR(r0,rt)] +                         i = 0, j = 1, k = t
        [σS(s0,st) - σS(s0,s1)]*[σR(r0,rt)-σR(r0,r1)] +                         i = 0, j = t, k = 1
                         
        [σS(s1,s0) - σS(s1,s1)]*[σR(r1,r0)-σR(r1,r1)] +                         i = 1, j = 0, k = 1
        [σS(s1,s1) - σS(s1,s0)]*[σR(r1,r1)-σR(r1,r0)] +                         i = 1, j = 1, k = 0
        [σS(s1,s0) - σS(s1,st)]*[σR(r1,r0)-σR(r1,rt)] +                         i = 1, j = 0, k = t
        [σS(s1,st) - σS(s1,s0)]*[σR(r1,rt)-σR(r1,r0)] +                         i = 1, j = t, k = 0
        [σS(s1,s1) - σS(s1,st)]*[σR(r1,r1)-σR(r1,rt)] +                         i = 1, j = 1, k = t
        [σS(s1,st) - σS(s1,s1)]*[σR(r1,rt)-σR(r1,r1)] +                         i = 1, j = t, k = 1
                         
        [σS(st,s0) - σS(st,s1)]*[σR(rt,r0)-σR(rt,r1)] +                         i = t, j = 0, k = 1
        [σS(st,s1) - σS(st,s0)]*[σR(rt,r1)-σR(rt,r0)] +                         i = t, j = 1, k = 0
        [σS(st,s0) - σS(st,st)]*[σR(rt,r0)-σR(rt,rt)] +                         i = t, j = 0, k = t
        [σS(st,st) - σS(st,s0)]*[σR(rt,rt)-σR(rt,r0)] +                         i = t, j = t, k = 0
        [σS(st,s1) - σS(st,st)]*[σR(rt,r1)-σR(rt,rt)] +                         i = t, j = 1, k = t
        [σS(st,st) - σS(st,s1)]*[σR(rt,rt)-σR(rt,r1)]                           i = t, j = t, k = 1
    )
    = 27/2 - (       by symmetry of the similarity
        [σS(s0,s0) - σS(s0,s1)]*[σR(r0,r0)-σR(r0,r1)] +                         i = 0, j = 0, k = 1
        [σS(s0,s0) - σS(s0,st)]*[σR(r0,r0)-σR(r0,rt)] +                         i = 0, j = 0, k = t
        [σS(s0,s1) - σS(s0,st)]*[σR(r0,r1)-σR(r0,rt)] +                         i = 0, j = 1, k = t
                         
        [σS(s1,s0) - σS(s1,s1)]*[σR(r1,r0)-σR(r1,r1)] +                         i = 1, j = 0, k = 1
        [σS(s1,s0) - σS(s1,st)]*[σR(r1,r0)-σR(r1,rt)] +                         i = 1, j = 0, k = t
        [σS(s1,s1) - σS(s1,st)]*[σR(r1,r1)-σR(r1,rt)] +                         i = 1, j = 1, k = t
                         
        [σS(st,s0) - σS(st,s1)]*[σR(rt,r0)-σR(rt,r1)] +                         i = t, j = 0, k = 1
        [σS(st,s0) - σS(st,st)]*[σR(rt,r0)-σR(rt,rt)] +                         i = t, j = 0, k = t
        [σS(st,s1) - σS(st,st)]*[σR(rt,r1)-σR(rt,rt)]                           i = t, j = 1, k = t
    )
    = 27/2 - (       
        [1 - σS(s0,s1)]*[1-0] +                         i = 0, j = 0, k = 1      
        [σS(s1,s0) - 1]*[0-1] +                         i = 1, j = 0, k = 1

        [σS(st,s0) - σS(st,s1)]*[σR(rt,r0)-σR(rt,r1)] +                         i = t, j = 0, k = 1
        [σS(s0,s1) - σS(s0,st)]*[0-σR(r0,rt)] +                         i = 0, j = 1, k = t
        [σS(s1,s0) - σS(s1,st)]*[0-σR(r1,rt)] +                         i = 1, j = 0, k = t
        [1 - σS(s1,st)]*[1-σR(r1,rt)] +                         i = 1, j = 1, k = t         
        [1 - σS(s0,st)]*[1-σR(r0,rt)] +                         i = 0, j = 0, k = t
        [σS(st,s0) - 1]*[σR(rt,r0)-1] +                         i = t, j = 0, k = t
        [σS(st,s1) - 1]*[σR(rt,r1)-1]                           i = t, j = 1, k = t
    )
    = 27/2 - (       
        2 * [1 - σS(s0,s1)]*[1-0] +                         i = 0, j = 0, k = 1 & i = 1, j = 0, k = 1

        [σS(st,s0) - σS(st,s1)]*[σR(rt,r0)-σR(rt,r1)] +     i = t, j = 0, k = 1
        [σS(s0,s1) - σS(s0,st)]*[0-σR(r0,rt)] +             i = 0, j = 1, k = t
        [σS(s0,s1) - σS(s1,st)]*[0-σR(r1,rt)] +             i = 1, j = 0, k = t
        2 * [1 - σS(s1,st)]*[1-σR(r1,rt)] +                 i = 1, j = 1, k = t & i = t, j = 1, k = t
        2 * [1 - σS(s0,st)]*[1-σR(r0,rt)]                   i = 0, j = 0, k = t & i = t, j = 0, k = t
    )

    if σR(r1,rt) = 1 (i.e. rt = r1)
    Γ(σS,σR,CB∪{(st,rt)}) = 27/2 - (       
        2 * [1 - σS(s0,s1)]*[1-0] +                         i = 0, j = 0, k = 1 & i = 1, j = 0, k = 1

        [σS(st,s0) - σS(st,s1)]*[0-1] +     i = t, j = 0, k = 1
        [σS(s0,s1) - σS(s0,st)]*[0-0] +             i = 0, j = 1, k = t
        [σS(s0,s1) - σS(s1,st)]*[0-1] +             i = 1, j = 0, k = t
        2 * [1 - σS(s1,st)]*[1-1] +                 i = 1, j = 1, k = t & i = t, j = 1, k = t
        2 * [1 - σS(s0,st)]*[1-0]                   i = 0, j = 0, k = t & i = t, j = 0, k = t
    )
    = 27/2 - (       
        2 * [1 - σS(s0,s1)] +                         i = 0, j = 0, k = 1 & i = 1, j = 0, k = 1
        -[σS(s0,st) - σS(s1,st)] +                    i = t, j = 0, k = 1
        -[σS(s0,s1) - σS(s1,st)] +                    i = 1, j = 0, k = t
        2 * [1 - σS(s0,st)]                           i = 0, j = 0, k = t & i = t, j = 0, k = t
    )
    = 27/2 - (       
        2 - 2 * σS(s0,s1) +                           i = 0, j = 0, k = 1 & i = 1, j = 0, k = 1
        -σS(s0,st) + σS(s1,st)] +                    i = t, j = 0, k = 1
        -σS(s0,s1) + σS(s1,st)] +                    i = 1, j = 0, k = t
        2 - 2 * σS(s0,st)                            i = 0, j = 0, k = t & i = t, j = 0, k = t
    )
    = 27/2 - 4 - 3 * σS(s0,s1) + 2 * σS(s1,st)] - 3 * σS(s0,st)
    = 19/2 - 3 * σS(s0,s1) + 2 * σS(s1,st)] - 3 * σS(s0,st)

    and the total energy is:
    
    E CoAT(x,y)=Γ(σS,σR,CB∪{(x,y)}) - Γ(σS,σR,CB) 
    = (27/2 - 4 - 3 * σS(s0,s1) + 2 * σS(s1,st)] - 3 * σS(s0,st)) - (2 + 2*σS(s0,s1))
    = (27/2 - 4 - 3 * σS(s0,s1) + 2 * σS(s1,st)] - 3 * σS(s0,st)) - 2 - 2*σS(s0,s1)
    = 27/2 - 6 - 5 * σS(s0,s1) + 2 * σS(s1,st)] - 3 * σS(s0,st)
    = 7.5 - 5 * σS(s0,s1) + 2 * σS(s1,st)] - 3 * σS(s0,st)
    """
    CtCoAT_energy_fadi = lambda x, y: (
        ((7.5 + 3*euclidean_sim(x,x2=X_cb[1]) - 2*euclidean_sim(x,X_cb[0]) + euclidean_sim(X_cb[0],X_cb[1])) / 27)
        if y==1 else
        ((7.5 + 3*euclidean_sim(x,X_cb[0]) - 2*euclidean_sim(x,X_cb[1]) + euclidean_sim(X_cb[0],X_cb[1])) / 27)
    )
    CtCoAT_energy_esteban = lambda x, y: (
        ((9.5 - 5*euclidean_sim(X_cb[0],x2=X_cb[1]) + 2*euclidean_sim(X_cb[1],x) - 3 * euclidean_sim(X_cb[0],x)))
        if y==1 else
        ((9.5 - 5*euclidean_sim(X_cb[0],x2=X_cb[1]) + 2*euclidean_sim(X_cb[0],x) - 3 * euclidean_sim(X_cb[1],x)))
    )
    def CtCoAT_gamma(simS, simR, X_cb, y_cb):
        # Γ(σS,σR,CB)= ∑_(i,j,k) (1 - [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)])/2

        sum_result = 0
        for X_i, y_i in zip(X_cb, y_cb):
            for X_j, y_j in zip(X_cb, y_cb):
                for X_k, y_k in zip(X_cb, y_cb):
                    sum_result += (1 - (simS(X_i, X_j) - simS(X_i, X_k)) * (simR(y_i, y_j) - simR(y_i, y_k)))/2
        return sum_result
    def CtCoAT_energy(X_, y_):
        Gamma_CB = CtCoAT_gamma(euclidean_sim, class_equality_sim, X_cb, y_cb)
        Gamma_CB_t = CtCoAT_gamma(euclidean_sim, class_equality_sim, 
                                  np.append(X_cb, values=X_.reshape(1, len(X_)), axis=0), np.append(y_cb, y_.reshape(1), axis=0))
        return Gamma_CB_t - Gamma_CB

    CtCoAT_hinge_fadi = lambda x, y, hinge_margin_: max(0,hinge_margin_ - 5*(euclidean_sim(x,X_cb[1]) - euclidean_sim(x,X_cb[0]))*(class_equality_sim(y,y_cb[1]) - class_equality_sim(y,y_cb[0])))

    manual_energy_ = {**manual_energy_,
        "energy_cb": 2 + np.exp(-2)*2,
        "epsilon": 1e-6,
        "energy_case_new": [
            CtCoAT_energy(X_, y_)
            for X_, y_ in zip(X, y)
        ],
        "hinge_case_new": {hinge_margin_*1e-3: [
            CtCoAT_hinge_fadi(X_, y_, hinge_margin_*1e-3)
            for X_, y_ in zip(X, y)
        ] for hinge_margin_ in range(1,10)},
    }

    return [manual_energy_]