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

from .energy_knn_backend import KNNEnergyComputations
from ..utils import to_numpy_array, pairwise_dist, cart_dist
from ..torch_utils import remove_index, append_symmetric
from .AbstractEnergyBasedClassifier import ACaseBaseEnergyClassifier, SourceSpaceElement, OutcomeSpaceElement

NumberOrBool = Union[float, int, bool]

class EnergyKNN(ACaseBaseEnergyClassifier):
    """
    
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
    """
    X_sim_matrix_ = None # float [|_X|, |_X|]
    y_sim_matrix_ = None # float [|_X|, |_X|]
    y_sim_vectors_ = None # float [|classes_|, |_X|], one vector per possible label
    device_ = None

    sim_X: Callable[[SourceSpaceElement, SourceSpaceElement], float]
    sim_y: Callable[[OutcomeSpaceElement, OutcomeSpaceElement], float]
    precompute_sim_matrix: bool
    
    def __init__(self,
                 sim_X: Callable[[SourceSpaceElement, SourceSpaceElement], float],
                 sim_y: Callable[[OutcomeSpaceElement, OutcomeSpaceElement], float],
                 n_neighbors: int = 3,
                 precompute_sim_matrix: bool= False):
        """
        Parameters
        ----------
        sim_X : the similarity measure for the source space
        sim_y : the similarity measure for the outcome space
        k : int (default=3)
            the number of neighbors to consider
        precompute_sim_matrix : bool (default=False)
            whether to compute the similarity matrices and vector during .fit or delay until first calls to energy_cb
        """
        try: pickle.dumps(sim_X)
        except AttributeError: raise ValueError("sim_X not pickleable, but it should be") 
        try: pickle.dumps(sim_y)
        except AttributeError: raise ValueError("sim_y not pickleable, but it should be") 
        self.sim_X = sim_X
        self.sim_y = sim_y
        self.n_neighbors = n_neighbors
        self.precompute_sim_matrix = precompute_sim_matrix
        
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

        return self

    def remove(self, index: int) -> EnergyKNN:
        """Returns a copy of this MeATCubeCB object where case `index` has been removed.
        
        If initialized, will copy and update the similarity matrices and the cube."""
        check_is_fitted(self)
        if isinstance(index, torch.Tensor): # failsafe
            index_cpu=int(index.cpu().item())
        else:
            index_cpu=index
        updated_knn = EnergyKNN(sim_X=self.sim_X,sim_y=self.sim_y)
        updated_knn.fit(
            X=np.delete(self._X, index_cpu, axis=0),
            y=np.delete(self._y, index_cpu, axis=0),
            classes=self.classes_,
            device=self.device_)

        # Copy the similarity matrices without the row at index nor the row at index (if already initialized)
        if self.X_sim_matrix_ is not None:
            updated_knn.X_sim_matrix_ = remove_index(self.X_sim_matrix_, index, dims=[-1,-2])
        if self.y_sim_matrix_ is not None:
            updated_knn.y_sim_matrix_ = remove_index(self.y_sim_matrix_, index, dims=[-1,-2])
        if self.y_sim_vectors_ is not None:
            updated_knn.y_sim_vectors_ = remove_index(self.y_sim_vectors_, index, dims=[-1])
        
        check_is_fitted(updated_knn)
        return updated_knn
    
    def add(self, case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement) -> EnergyKNN:
        """Returns a copy of this MeATCubeCB object where case `index` has been removed.
        
        If initialized, will copy and update the similarity matrices and the cube."""
        check_is_fitted(self)
        updated_knn = EnergyKNN(sim_X=self.sim_X,sim_y=self.sim_y)
        updated_knn.fit(
            X=np.append(self._X, [case_source], axis=0),
            y=np.append(self._y, [case_outcome], axis=0),
            classes=self.classes_,
            device=self.device_)
        
        # Extend the similarity matrix with the new similarity (if already initialized)
        if self.X_sim_matrix_ is not None:
            source_sim_vect = self._source_sim_vect(case_source)
            source_sim_reflexive = torch.tensor(self.sim_X(case_source, case_source), device=self.device_)
            updated_knn.X_sim_matrix_ = append_symmetric(
                self.X_sim_matrix_, source_sim_vect, source_sim_reflexive.view(-1))
        if self.y_sim_matrix_ is not None:
            outcome_sim_vect = self._outcome_sim_vect(case_outcome)
            outcome_sim_reflexive = torch.tensor(self.sim_y(case_outcome, case_outcome), device=self.device_)
            updated_knn.y_sim_matrix_ = append_symmetric(
                self.y_sim_matrix_, outcome_sim_vect, outcome_sim_reflexive.view(-1))
        
        check_is_fitted(updated_knn)
        return updated_knn
    
    def to_device(self, device: Literal["auto"] | str | torch.device = "auto"):
        if device == "auto":
            if torch.cuda.is_available():
                try: 
                    torch.randn((5,10,), device="cuda")
                    self.device_ = torch.device("cuda")
                except RuntimeError:
                    # TODO: add log message
                    self.device_ = torch.device("cpu")
            else:
                self.device_ = torch.device("cpu")
        else:
            self.device_ = device

        for param in [self.X_sim_matrix_, self.y_sim_matrix_, self.y_sim_vectors_]:
            if param is not None:
                param = param.to(self.device_)

        return self.device_


###########################################################
# TODO: check output shape

    def energy_cb(self, as_tensor=False):
        self._compute_sim_matrix()

        # X_sim_matrix_, but with -inf in the diagonal to prevent adding
        no_equality_X_sim_matrix_ = torch.masked_fill(self.X_sim_matrix_, torch.eye(self.X_sim_matrix_.size(0)), -torch.inf)
        no_equality_y_sim_matrix_ = torch.masked_fill(self.y_sim_matrix_, torch.eye(self.X_sim_matrix_.size(0)), torch.nan)
        #knn_mask = KNNEnergyComputations.knn_mask(no_equality_X_sim_matrix_)
        energy = KNNEnergyComputations.energy_zip_matrix(no_equality_X_sim_matrix_, no_equality_y_sim_matrix_, k=self.n_neighbors)
        if as_tensor: return energy
        return energy.cpu().item()
    
    def energy_case_from_cb(self, index: int, as_tensor=False):
        # X_sim_matrix_, but with -inf in the diagonal to prevent adding
        no_equality_X_sim_matrix_ = torch.masked_fill(self.X_sim_matrix_, torch.eye(self.X_sim_matrix_.size(0)), -torch.inf)[:,index]
        no_equality_y_sim_matrix_ = torch.masked_fill(self.y_sim_matrix_, torch.eye(self.X_sim_matrix_.size(0)), torch.nan)[:,index]
        #knn_mask = KNNEnergyComputations.knn_mask(no_equality_X_sim_matrix_)
        energy = KNNEnergyComputations.energy_zip_matrix(no_equality_X_sim_matrix_, no_equality_y_sim_matrix_, k=self.n_neighbors)
        if as_tensor: return energy
        return energy.cpu().item()
    
    def energy_case_new(self, X: SourceSpaceElement, y: OutcomeSpaceElement, as_tensor=False) -> float:
        self._compute_sim_matrix()
        self._compute_outcome_sim_vectors()

        # computes the similarity of the new case to the ones in the CB
        X_sim_vectors = self._source_sim_vect(X).transpose(-1,-2)
        label_index = self._outcome_index(y)
        y_sim_vectors = self.y_sim_vectors_.select(-2, label_index).transpose(-1,-2)

        energy = KNNEnergyComputations.energy_zip_matrix(X_sim_vectors, y_sim_vectors, k=self.n_neighbors)
        if as_tensor: return energy
        return energy.cpu().item()
    def energy_cases_new(self,
                         X: Iterable[SourceSpaceElement],
                         y: Iterable[OutcomeSpaceElement], as_tensor=False) -> torch.FloatTensor:
        self._compute_sim_matrix()
        self._compute_outcome_sim_vectors()

        # computes the similarity of the new case to the ones in the CB
        X_sim_vectors = self._source_sim_vect(X).transpose(-1,-2)
        y_sim_vectors = self._outcome_sim_vect(y).transpose(-1,-2)

        energy = KNNEnergyComputations.energy_map_matrix(X_sim_vectors, y_sim_vectors, k=self.n_neighbors)
        if as_tensor: return energy
        return energy.cpu().item()


    def decrement_scores(self,
            X_ref: Iterable[SourceSpaceElement],
            y_ref: Iterable[OutcomeSpaceElement],
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            as_tensor=False,
            **kwargs) -> np.ndarray[float]:
        self._compute_sim_matrix()
        self._compute_outcome_sim_vectors()

        # energy for full CB (ommitable): [|S|, |R|]
        source_sim_vectors = self._source_sim_vect(X_ref).T # [|CB|, |S|]
        energies_cb = KNNEnergyComputations.energy_map_matrix(
            source_sim_vectors, 
            self.y_sim_vectors_.T) # [|S|, |R|]
        
        # --------- All the cases at once ---------
        index = list(range(len(self)))
        try:
            batch_size = len(index)
            energies_i = self._decrement_scores_batched(
                index=index,
                source_sim_vectors=source_sim_vectors,
                batch_size=batch_size,
                auto_batch_size=False
            )
        except torch.cuda.OutOfMemoryError:
            batch_size = 1
            # batch_size_power = int(np.floor(np.log2(len(index))).item())
            # batch_size = 2**batch_size_power
            energies_i = self._decrement_scores_batched(
                index=index,
                source_sim_vectors=source_sim_vectors,
                batch_size=batch_size,
                auto_batch_size=False
            ) # [|CB|, |S|, |R|]

        # compare the true outcome with the other outcomes
        true_outcome_index = self._outcome_index(y_ref)
        mask = torch.arange(
            energies_cb.size(-1),
            device=energies_cb.device
        ).unsqueeze(0) == true_outcome_index.unsqueeze(1)
        l_mce = energies_cb[~mask].min(dim=-1).values - energies_cb[mask]
        mask_max = (mask.unsqueeze(0)) * (energies_i.max().detach() + 1) # trick to "exclude" the mask from the min
        l_mce_i = (energies_i + mask_max).min(dim=-1).values - energies_i[:,mask]
        # l_mce: [|S|]
        # l_mce_i: [|index|, |S|]

        # if hinge loss, modify a bit before aggregation
        if strategy=="hinge":
            l = -(margin - l_mce).clamp(min=0)
            l_i = -(margin - l_mce_i).clamp(min=0)
        else:
            l = l_mce
            l_i = l_mce_i
        # l: [|S|]
        # l_i: [|index|, |S|]

        l = l.unsqueeze(0) - l_i 
        # l: [|index|, |S|]

        # aggregate the results
        # l: if aggregation is None or "none": [|index|, |S|]
        # l: otherwise: [|index|]
        l = l.mean(dim=-1)

        if as_tensor: return l
        return l.cpu().numpy()

    def _decrement_scores_batched(self,
                                  index,
                                source_sim_vectors: Iterable[SourceSpaceElement],
                                batch_size,
                                auto_batch_size=True,
                                **kwargs) -> np.ndarray[float]:
        try:
            inversion_rates_i = []
            index_batches = [index[i:i+batch_size] for i in range(0, len(index), batch_size)]
            for index_batch in index_batches:
                # we need for every i: 
                # - the similarity between CB_X/i and X_ref
                # - the similarity between CB_y/i and known labels
                sim_S_i = torch.stack([
                        remove_index(source_sim_vectors, i, dims=[-2])
                        for i in index_batch
                    ], dim=0).detach()
                sim_R_i = torch.stack([
                        remove_index(self.y_sim_vectors_.T, i, dims=[-2])
                        for i in index_batch
                    ], dim=0).detach()
        
                # inversion_rates_i: [|index_batch|, |S|, |R|]
                inversion_rates_i.append(
                    KNNEnergyComputations.energy_map_matrix(
                        sim_S_i,
                        sim_R_i))
            inversion_rates_i = torch.cat(inversion_rates_i, dim=0)
        
        except torch.cuda.OutOfMemoryError as e:
            if auto_batch_size and batch_size > 1:
                inversion_rates_i = self._decrement_scores_batched(
                                    index,
                                    X_ref,
                                    batch_size=batch_size//2,
                                    auto_batch_size=auto_batch_size,
                                    **kwargs)
                self.inferred_batch_size_ = batch_size # update the batch sisze
                return inversion_rates_i
            else: raise e
        return inversion_rates_i



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