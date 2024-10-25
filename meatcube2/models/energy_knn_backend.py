
# %%
import torch
import numpy as np
import pandas as pd
from typing import Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List
from scipy.spatial.distance import squareform, pdist, cdist
from tqdm.auto import tqdm
#from ..torch_utils import pop_index

# %%
class KNNEnergyComputations(object):
    @staticmethod
    def energy_map_matrix(sim_S: torch.Tensor, sim_R: torch.Tensor, k=3):
        """Energy of the CB for each source and outcome combination.

        sim_S[*, i, j] corresponds to case i from the CB and case source j to predict.
        sim_R[*, i, k] corresponds to case i from the CB and case outcome k to predict.
        
        The output is an energy matrix where energy[*, j, k] corresponds to the energy of the CB for case source j and
        case outcome k.
        """
        knn_mask = KNNEnergyComputations.knn_mask(sim_S, k) # [|CB|, N] 
        energy = 1-(knn_mask.unsqueeze(-1) * sim_R.unsqueeze(-2)).sum(-3)/min(k, sim_S.shape[-2]) # [N, M]
        return energy
        
    @staticmethod
    def energy_map_value(CB_X: torch.Tensor, CB_y: torch.Tensor, X: torch.Tensor, y: torch.Tensor, k=3):
        """Energy of the CB for each source and outcome combination.

        CB_X[*, i] and CB_y[*, i] correspond to the source and outcome of case i from the CB. 
        X[*, j] corresponds to case source j to predict.
        y[*, k] corresponds to case outcome k to predict.
        
        The output is an energy matrix where energy[*, j, k] corresponds to the energy of the CB for case source j and
        case outcome k.
        """
        sim_S = KNNEnergyComputations.sim_matrix(CB_X, X) # [|CB|, N]
        sim_R = KNNEnergyComputations.sim_matrix(CB_y, y) # [|CB|, M]
        return KNNEnergyComputations.energy_map_matrix(sim_S, sim_R, k=k)
    
    @staticmethod
    def energy_zip_matrix(sim_S: torch.Tensor, sim_R: torch.Tensor, k=3):
        """Energy of the CB for each case source and outcome combination.

        sim_S[*, i, j] and sim_R[*, i, j] correspond to case i from the CB and case j to predict.
        
        The output is an energy matrix where energy[*, j] corresponds to the energy of the CB for case j.
        """
        knn_mask = KNNEnergyComputations.knn_mask(sim_S, k)
        energy = 1-(knn_mask * sim_R).sum(-3)/min(k, sim_S.shape[-2]) # [N]
        return energy
    
    @staticmethod
    def energy_zip_value(CB_X: torch.Tensor, CB_y: torch.Tensor, X: torch.Tensor, y: torch.Tensor, k=3):
        """Energy of the CB for each source and outcome combination.

        CB_X[*, i] and CB_y[*, i] correspond to the source and outcome of case i from the CB. 
        X[*, j] and y[*, j] correspond to case j to predict.
        
        The output is an energy matrix where energy[*, j] corresponds to the energy of the CB for case j.
        """
        sim_S = KNNEnergyComputations.sim_matrix(CB_X, X) # [|CB|, N]
        sim_R = KNNEnergyComputations.sim_matrix(CB_y, y) # [|CB|, N]
        return KNNEnergyComputations.energy_zip_matrix(sim_S, sim_R, k=k)

    @staticmethod
    def sim_matrix(CB_X, X):
        dists = -torch.pairwise_distance(X.unsqueeze(0), CB_X.unsqueeze(1)) # [|CB|, N]
        return dists

    @staticmethod
    def knn_mask(sim_matrix, k=3):
        # dists: [|CB|, N]
        k_neighboors = torch.argsort(sim_matrix, dim=0, descending=True)[:k] # [K, N]
        
        label_mask = (
            torch.arange(sim_matrix.size(0), device=sim_matrix.device).unsqueeze(1).unsqueeze(2) # [|CB|, *]
            == 
            k_neighboors) # [|CB|, K, N]
        label_mask = label_mask.any(dim=1)# [|CB|, N]
        return label_mask

# %%
if __name__ == "__main__":
    
    N   =5
    CB_N=10
    d=2
    K = 3

    X = torch.randn((N, d,)) # target points [N, d]
    CB_X = torch.randn((CB_N, d,)) # target points [|CB|, d]
    CB_y = torch.randint(0,4,(CB_N,))

    dists = torch.pairwise_distance(X.unsqueeze(0), CB_X.unsqueeze(1)) # [|CB|, N]
    k_neighboors = torch.argsort(dists, dim=0)[:K] # [K, N]
    dists.size(), k_neighboors.size(), k_neighboors
    #%%
    label_mask = (torch.arange(CB_y.size(0)).unsqueeze(1).unsqueeze(2) # [|CB|, *]
        == 
        k_neighboors) # [|CB|, K, N]
    label_mask = label_mask.any(dim=1) # [|CB|, N]

    labels = (CB_y.unsqueeze(1) * label_mask)
    # %%
    k_neighboors_labels = CB_y[k_neighboors.view(-1)].view(3,-1)
    k_neighboors_labels
    # %%
    k_neighboors_labels = (CB_y.unsqueeze(1).unsqueeze(2) * label_mask).sum(dim=0) # [K, N]
    k_neighboors_labels
    # %%
    torch.mode(k_neighboors_labels, dim=0).values
    # %%
        