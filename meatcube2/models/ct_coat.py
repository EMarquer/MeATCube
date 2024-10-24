from __future__ import annotations # for self-referring type hints
import torch, numpy as np
import pandas as pd
from typing import Any, Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List
from typing_extensions import Self
from collections.abc import Sequence
from scipy.spatial.distance import squareform, pdist, cdist
from tqdm.auto import tqdm
import pickle
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from meatcube2.models.AbstractEnergyBasedPredictor import ACaseBaseEnergyPredictor

from .ct_coat_backend import CtCoATEnergyComputations
from ..torch_utils import remove_index, append_symmetric
from ..utils import to_numpy_array, pairwise_dist, cart_dist
from ..defaults import MEATCUBE_COMPETENCE_NORMALIZE_BY_MAX_COMPETENCE
from .AbstractEnergyBasedClassifier import ACaseBaseEnergyClassifier, SourceSpaceElement, OutcomeSpaceElement

from itertools import combinations,permutations
NumberOrBool = Union[float, int, bool]

class CtCoAT(ACaseBaseEnergyClassifier):
    """Collection of tensors and metrics that automate the computations of several metrics based on the number of 
    energies, and supports addition and deletion of cases.
    
    Based on MeATCube: (Me)asure of the complexity of a dataset for (A)nalogical (T)ransfer using Boolean (Cube)s, or 
    slices of them.
    
    Attributes
    ----------
    sim_X : (SourceSpaceElement,SourceSpaceElement) -> float
        the situation space similarity 
    sim_y : (OutcomeSpaceElement,OutcomeSpaceElement) -> float
        the outcome space similarity
    
    Inherited attributes
    ----------
    _X : Iterable[SourceSpaceElement]
        the situations of the cases in the CB
    _y : Iterable[OutcomeSpaceElement]
        the outcomes of the cases in the CB
    classes_ : List[OutcomeSpaceElement]
        the list of possible classes in the CB
    n_features_in_: int
        number of features expected in the situation space

    
    TODO
    ----
    optimize :     
        .decrement_scores
        
        .increment_scores
    """
    X_sim_matrix_ = None # float [|_X|, |_X|]
    y_sim_matrix_ = None # float [|_X|, |_X|]
    y_sim_vectors_ = None # float [|classes_|, |_X|], one vector per possible label
    cube_ = None # bool [|_X|, |_X|]
    device_ = None

    sim_X: Callable[[SourceSpaceElement, SourceSpaceElement], float]
    sim_y: Callable[[OutcomeSpaceElement, OutcomeSpaceElement], float]
    precompute_cube: bool
    precompute_sim_matrix: bool
    normalize=True
    
    def __init__(self,
                 sim_X: Callable[[SourceSpaceElement, SourceSpaceElement], float],
                 sim_y: Callable[[OutcomeSpaceElement, OutcomeSpaceElement], float],
                 precompute_cube: bool= False,
                 precompute_sim_matrix: bool= False,
                 normalize=True):
        """
        Parameters
        ----------
        sim_X : the similarity measure for the source space
        sim_y : the similarity measure for the outcome space
        precompute_cube : bool (default=False)
            whether to compute the cube during .fit or delay until first calls to energy_cb
        precompute_sim_matrix : bool (default=False)
            whether to compute the similarity matrices and vector during .fit or delay until first calls to energy_cb
        """
        try: pickle.dumps(sim_X)
        except AttributeError: raise ValueError("sim_X not pickleable, but it should be") 
        try: pickle.dumps(sim_y)
        except AttributeError: raise ValueError("sim_y not pickleable, but it should be") 
        self.sim_X = sim_X
        self.sim_y = sim_y
        self.precompute_cube = precompute_cube
        self.precompute_sim_matrix = precompute_sim_matrix
        self.normalize = normalize
        
    def fit(self, 
            X: Iterable[SourceSpaceElement],
            y: Iterable[OutcomeSpaceElement],
            classes: List[OutcomeSpaceElement] | Literal['infer'] = "infer",
            force_copy=True,
            device: Literal["auto"] | str | torch.device = "auto") -> Self:
        super().fit(X, y, classes, force_copy)

        self.X_sim_matrix_ = None # [|CB|, |CB|]
        self.y_sim_matrix_ = None # [|CB|, |CB|]
        self.y_sim_vectors_ = None # [|R|, |CB|], one vector per possible outcome
        self.cube_ = None # [|CB|, |CB|, |CB|]

        self.to_device(device)

        # add precomputed cube and/or similarity matrices
        if self.precompute_sim_matrix:
            self._compute_sim_matrix()
            self._compute_outcome_sim_vectors()
        if self.precompute_cube:
            self._compute_inversion_cube()

        return self

    def remove(self, index: int) -> CtCoAT:
        """Returns a copy of this MeATCubeCB object where case `index` has been removed.
        
        If initialized, will copy and update the similarity matrices and the cube."""
        check_is_fitted(self)
        if isinstance(index, torch.Tensor): # failsafe
            index_cpu=int(index.cpu().item())
        else:
            index_cpu=index
        updated_meatcube = CtCoAT(sim_X=self.sim_X,sim_y=self.sim_y)
        updated_meatcube.fit(
            X=np.delete(self._X, index_cpu, axis=0),
            y=np.delete(self._y, index_cpu, axis=0),
            classes=self.classes_,
            device=self.device_)

        # Copy the similarity matrices without the row at index nor the row at index (if already initialized)
        if self.X_sim_matrix_ is not None:
            updated_meatcube.X_sim_matrix_ = remove_index(self.X_sim_matrix_, index, dims=[-1,-2])
        if self.y_sim_matrix_ is not None:
            updated_meatcube.y_sim_matrix_ = remove_index(self.y_sim_matrix_, index, dims=[-1,-2])
        if self.y_sim_vectors_ is not None:
            updated_meatcube.y_sim_vectors_ = remove_index(self.y_sim_vectors_, index, dims=[-1])
        if self.cube_ is not None:
            updated_meatcube.cube_ = remove_index(self.cube_, index, dims=[-1,-2,-3])
        
        check_is_fitted(updated_meatcube)
        return updated_meatcube
    
    def add(self, case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement) -> CtCoAT:
        """Returns a copy of this MeATCubeCB object where case `index` has been removed.
        
        If initialized, will copy and update the similarity matrices and the cube."""
        check_is_fitted(self)
        updated_meatcube = CtCoAT(sim_X=self.sim_X,sim_y=self.sim_y)
        updated_meatcube.fit(
            X=np.append(self._X, [case_source], axis=0),
            y=np.append(self._y, [case_outcome], axis=0),
            classes=self.classes_,
            device=self.device_)
        
        # Extend the similarity matrix with the new similarity (if already initialized)
        if self.X_sim_matrix_ is not None:
            source_sim_vect = self._source_sim_vect(case_source)
            source_sim_reflexive = torch.tensor(self.sim_X(case_source, case_source), device=self.device_)
            updated_meatcube.X_sim_matrix_ = append_symmetric(
                self.X_sim_matrix_, source_sim_vect, source_sim_reflexive.view(-1))
        if self.y_sim_matrix_ is not None:
            outcome_sim_vect = self._outcome_sim_vect(case_outcome)
            outcome_sim_reflexive = torch.tensor(self.sim_y(case_outcome, case_outcome), device=self.device_)
            updated_meatcube.y_sim_matrix_ = append_symmetric(
                self.y_sim_matrix_, outcome_sim_vect, outcome_sim_reflexive.view(-1))
        
        # Extend the inversion cube with the new energies (if already initialized)
        if self.X_sim_matrix_ is not None and self.y_sim_matrix_ is not None and self.cube_ is not None:
            inv_ibc, inv_aic, inv_abi, inv_aii, inv_ibi, inv_iic, inv_iii = CtCoATEnergyComputations._energies_i(
                self.X_sim_matrix_, self.y_sim_matrix_, # [..., M, M]
                source_sim_vect, outcome_sim_vect, # [..., M]
                reflexive_sim_source=source_sim_reflexive, reflexive_sim_outcome=outcome_sim_reflexive, # [...] or []
                exclude_impossible=False)
            
            # from [n, n, n] to [n, n, n+1]
            updated_meatcube.cube_ = torch.cat([updated_meatcube.cube, inv_abi], dim=-1)

            # from [n, n].[n, 1] to [n, n+1]: add the symmetric component of the vector where the diagonal will be
            inv_aic = torch.cat([inv_aic, inv_aii.unsqueeze(-1)], dim=-1)
            # from [n, n, n+1].[n, n+1] to [n, n+1, n+1]
            updated_meatcube.cube_ = torch.cat([updated_meatcube.cube, inv_aic.unsqueeze(-2)], dim=-2)

            # from [n].[] to [n+1]
            inv_iic = torch.cat([inv_iic, inv_iii.unsqueeze(-1)], dim=-1)
            # from [n, n].[n, 1] to [n, n+1] to [n+1, n+1]
            inv_ibc = torch.cat([inv_ibc, inv_ibi.unsqueeze(-1)], dim=-1)
            inv_ibc = torch.cat([inv_ibc, inv_iic.unsqueeze(-2)], dim=-2)
            # from [n, n+1, n+1].[n+1, n+1] to [n+1, n+1, n+1]
            updated_meatcube.cube_ = torch.cat([updated_meatcube.cube, inv_ibc.unsqueeze(-3)], dim=-3)

        check_is_fitted(updated_meatcube)
        return updated_meatcube
    
    def to_device(self, device: Literal["auto"] | str | torch.device = "auto"):
        if device == "auto":
            if torch.cuda.is_available():
                self.device_ = torch.device("cuda")
            else:
                self.device_ = torch.device("cpu")
        else:
            self.device_ = device

        for param in [self.X_sim_matrix_, self.y_sim_matrix_, self.y_sim_vectors_, self.cube_]:
            if param is not None:
                param = param.to(self.device_)

        return self.device_



###########################################################
# TODO: check output shape

    def energy_cb(self, as_tensor=False):
        # self._compute_sim_matrix()
        
        # e = 0
        # for i in range(len(self)):
        #     for (j,k) in combinations(range(len(self)),2):
        #         e += (1 - (self.X_sim_matrix_[i,j]-self.X_sim_matrix_[i,k])*(self.y_sim_matrix_[i,j]-self.y_sim_matrix_[i,k]))

        # e = (e + (len(self)^2)/2)/(len(self)^3)

        # if as_tensor: return e
        # return e.cpu().item()
        self._compute_inversion_cube()
        energies = CtCoATEnergyComputations._energy(self.cube_)
        if self.normalize: energies = CtCoATEnergyComputations.normalize(energies, len(self))
        if as_tensor: return energies
        return energies.cpu().item()
    def energy_case_from_cb(self, index: int, as_tensor=False):
        if self.cube_ is not None:
            energies = CtCoATEnergyComputations._cube_gamma_i_included(self.cube_, index)
        else:
            self._compute_sim_matrix()
            energies = CtCoATEnergyComputations._gamma_i_included(self.X_sim_matrix_, self.y_sim_matrix_, index)
        if self.normalize: energies = CtCoATEnergyComputations.normalize(energies, len(self))
        if as_tensor: return energies
        return energies.cpu().item()
    def energy_case_new(self, X: SourceSpaceElement, y: OutcomeSpaceElement, as_tensor=False) -> float:
        self._compute_sim_matrix()
        self._compute_outcome_sim_vectors()

        # computes the similarity of the new case to the ones in the CB
        X_sim_vectors = self._source_sim_vect(X)
        label_index = self._outcome_index(y)
        y_sim_vectors = self.y_sim_vectors_.select(-2, label_index)
        reflexive_sim_X = self._source_sim_reflexive(X)
        reflexive_sim_y = self._outcome_sim_reflexive(y)

        energies = CtCoATEnergyComputations._gamma_i(self.X_sim_matrix_,
                                 self.y_sim_matrix_,
                                 X_sim_vectors,
                                 y_sim_vectors,
                                 reflexive_sim_source=reflexive_sim_X,
                                 reflexive_sim_outcome=reflexive_sim_y)
        if self.normalize: energies = CtCoATEnergyComputations.normalize(energies, len(self))
        if as_tensor: return energies
        return energies.cpu().item()
    def energy_cases_new(self,
                         X: Iterable[SourceSpaceElement],
                         y: Iterable[OutcomeSpaceElement], as_tensor=False) -> torch.FloatTensor:
        self._compute_sim_matrix()
        self._compute_outcome_sim_vectors()

        # computes the similarity of the new case to the ones in the CB
        X_sim_vectors = self._source_sim_vect(X)
        y_sim_vectors = self._outcome_sim_vect(y)
        reflexive_sim_X = self._source_sim_reflexive(X)
        reflexive_sim_y = self._outcome_sim_reflexive(y)
        
        # add dummy dimensions for broadcasting
        X_sim_vectors = X_sim_vectors.unsqueeze(-2)
        y_sim_vectors = y_sim_vectors.unsqueeze(-3)
        reflexive_sim_X = reflexive_sim_X.unsqueeze(-1)
        reflexive_sim_y = reflexive_sim_y.unsqueeze(-2)

        energies = CtCoATEnergyComputations._gamma_i(self.X_sim_matrix_,
                                 self.y_sim_matrix_,
                                 X_sim_vectors,
                                 y_sim_vectors,
                                 reflexive_sim_source=reflexive_sim_X,
                                 reflexive_sim_outcome=reflexive_sim_y)
        if self.normalize: energies = CtCoATEnergyComputations.normalize(energies, len(self))
        if as_tensor: return energies
        return energies.cpu().tolist()
    

    def predict_proba(self, X: Iterable[SourceSpaceElement]) -> Iterable[Iterable[float]]:
        # Check if fit has been called
        check_is_fitted(self)
        check_array(X,  accept_sparse=True, dtype=None, ensure_2d=False, allow_nd=False)
        
        energies = self.energy_cases_new(X, self.classes_, as_tensor=True)
        if self.normalize: energies = CtCoATEnergyComputations.normalize(energies, len(self))
        return (-energies.to(dtype=float)).softmax(-1).cpu().numpy()
        #return 1 - ((energies.float()-energies.min(dim=-1,keepdim=True).values) / (energies.max(dim=-1,keepdim=True).values-energies.min(dim=-1,keepdim=True).values)).numpy()
    
###########################################################
    
    
    def _is_source_list(self, value: Union[SourceSpaceElement, Iterable[SourceSpaceElement]]) -> bool:
        """Returns true if `value` is Iterable[SourceSpaceElement]], false otherwise."""
        if isinstance(value, torch.Tensor):
            return value.dim() == self._X.ndim
        elif isinstance(value, np.ndarray) or isinstance(value, (pd.DataFrame, pd.Series)):
            return value.ndim == self._X.ndim
        elif isinstance(value, str):
            return False
        elif isinstance(value, Iterable):
            return (self._X.ndim <= 1) or isinstance(value[0], Iterable)
        else:
            return False
    def _is_outcome_list(self, value: Union[OutcomeSpaceElement, Iterable[OutcomeSpaceElement]]) -> bool:
        if isinstance(value, torch.Tensor):
            return value.dim() == self._y.ndim
        elif isinstance(value, np.ndarray) or isinstance(value, (pd.DataFrame, pd.Series)):
            return value.ndim == self._y.ndim
        elif isinstance(value, str):
            return False
        elif isinstance(value, Iterable):
            return (self._y.ndim <= 1) or isinstance(value[0], Iterable)
        else:
            return False
    def _prep_source_for_dist(self, value: Union[SourceSpaceElement, Iterable[SourceSpaceElement]]) -> bool:
        if hasattr(self._X[0], "size"):
            return np.array(value).reshape(-1,self._X[0].size)
        else:
            return np.array(value).reshape(-1,1)
    def _prep_outcome_for_dist(self, value: Union[SourceSpaceElement, Iterable[SourceSpaceElement]]) -> bool:
        if hasattr(self._y[0], "size"):
            return np.array(value).reshape(-1,self._y[0].size)
        else:
            return np.array(value).reshape(-1,1)

    def _outcome_index(self, outcome: Union[OutcomeSpaceElement, Iterable[OutcomeSpaceElement]]) -> torch.LongTensor:
        if isinstance(self.classes_, np.ndarray):
            index = lambda x: np.where(self.classes_==x)[0][0]
            if self._is_outcome_list(outcome):
                return torch.tensor([index(o) for o in outcome]).to(self.device_)
            return torch.tensor(index(outcome)).to(self.device_)
        else:
            if self._is_outcome_list(outcome):
                return torch.tensor([self.classes_.index(o) for o in outcome]).to(self.device_)
            return torch.tensor(self.classes_.index(outcome)).to(self.device_)

    def _compute_outcome_sim_vectors(self, force_recompute: bool=False) -> None:
        """Computes the similarity vectors for each possible outcime."""
        if force_recompute or self.y_sim_vectors_ is None:
            potential = self._prep_outcome_for_dist(self.classes_)
            self.y_sim_vectors_ = torch.tensor(cdist(
                potential, self._prep_outcome_for_dist(self._y), metric=self.sim_y))
            self.y_sim_vectors_ = self.y_sim_vectors_.to(self.device_)

    def _compute_sim_matrix(self, force_recompute: bool=False) -> None:
        """Computes the similarity matrices."""
        if force_recompute or self.X_sim_matrix_ is None:
            self.X_sim_matrix_ = torch.tensor(squareform(pdist(self._prep_source_for_dist(self._X), metric=self.sim_X)))
            self.X_sim_matrix_ = self.X_sim_matrix_.diagonal_scatter(self._source_sim_reflexive(self._X))
            self.X_sim_matrix_ = self.X_sim_matrix_.to(self.device_)
        if force_recompute or self.y_sim_matrix_ is None:
            self.y_sim_matrix_ = torch.tensor(squareform(pdist(self._prep_outcome_for_dist(self._y), metric=self.sim_y)))
            self.y_sim_matrix_ = self.y_sim_matrix_.diagonal_scatter(self._outcome_sim_reflexive(self._y))
            self.y_sim_matrix_ = self.y_sim_matrix_.to(self.device_)

    def _compute_inversion_cube(self, force_recompute: bool=False) -> None:
        """Computes the inversion cube."""
        if force_recompute or self.cube_ is None:
            # we need the similarity matrices
            self._compute_sim_matrix(force_recompute=force_recompute)

            # then we compute the cube
            self.cube_ = CtCoATEnergyComputations._inversion_cube(self.X_sim_matrix_, self.y_sim_matrix_, return_all=False)

    def _source_sim_vect(self, sources: Union[SourceSpaceElement, Iterable[SourceSpaceElement]]) -> torch.Tensor:
        """Computes the similarity between the source and the CB in the source space.
        
        If `sources` is an iterable of size `|S|`, the result is of shape `[|S|, |CB|]`.
        If `sources` is a single value, the result is of shape `[|CB|]`.
        """
        sim_vect = torch.tensor(cdist(
            self._prep_source_for_dist(sources),
            self._prep_source_for_dist(self._X),
            metric=self.sim_X))
        
        return sim_vect.to(self.device_)
    def _source_sim_reflexive(self, sources: Union[SourceSpaceElement, Iterable[SourceSpaceElement]]) -> torch.Tensor:
        """Computes the similarity between the source and itself.
        
        If `sources` is an iterable of size `|S|`, the result is of shape `[|S|]`.
        If `sources` is a single value, the result is of shape `[]`.
        """
        sources = to_numpy_array(sources)
        if self._is_source_list(sources):
            sim_reflexive = torch.tensor([self.sim_X(source, source) for source in sources])
        else:
            sim_reflexive = torch.tensor(self.sim_X(sources, sources))
        
        return sim_reflexive.to(self.device_)
    
    def _outcome_sim_vect(self, outcomes: Union[OutcomeSpaceElement, Iterable[OutcomeSpaceElement]]) -> torch.Tensor:
        """Computes the similarity between the outcome and the CB in the outcome space.
        
        If `outcomes` is an iterable of size `|R|`, the result is of shape `[|R|, |CB|]`.
        If `outcomes` is a single value, the result is of shape `[|CB|]`.
        """
        sim_vect = torch.tensor(cdist(
            self._prep_outcome_for_dist(outcomes),
            self._prep_outcome_for_dist(self._y),
            metric=self.sim_y))
        
        return sim_vect.to(self.device_)
    def _outcome_sim_reflexive(self, outcomes: Union[OutcomeSpaceElement, Iterable[OutcomeSpaceElement]]) -> torch.Tensor:
        """Computes the similarity between the source and itself.
        
        If `outcomes` is an iterable of size `|R|`, the result is of shape `[|R|]`.
        If `outcomes` is a single value, the result is of shape `[]`.
        """
        if self._is_outcome_list(outcomes):
            sim_reflexive = torch.tensor([self.sim_y(outcome, outcome) for outcome in outcomes])
        else:
            sim_reflexive = torch.tensor(self.sim_y(outcomes, outcomes))
        
        return sim_reflexive.to(self.device_)
    




    #######################################################
    def _more_tags(self):
        return {
            "_xfail_checks": {
                # check_estimator checks that fail:
                "check_complex_data": "AssertionError: Did not raise: [<class 'TypeError'>]",
            },
            "allow_nan": False,
            "poor_score": True,
            "requires_y": True,
            "X_types": ["2darray", "sparse", "categorical", "1dlabels", "2dlabels", "string"]
        }