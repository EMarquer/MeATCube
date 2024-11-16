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
import logging
logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)

from meatcube2.models.AbstractEnergyBasedPredictor import ACaseBaseEnergyPredictor

from .ct_coat_backend import CtCoATEnergyComputations
from ..torch_utils import remove_index, append_symmetric
from ..utils import to_numpy_array, pairwise_dist, cart_dist, estimate_batch_size, estimate_mem_utilisation
from ..defaults import MEATCUBE_COMPETENCE_NORMALIZE_BY_MAX_COMPETENCE
from .AbstractEnergyBasedClassifier import ACaseBaseEnergyClassifier, SourceSpaceElement, OutcomeSpaceElement

from itertools import combinations,permutations
NumberOrBool = Union[float, int, bool]

TQDM_VERBOSE = False

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
        
        # # Extend the inversion cube with the new energies (if already initialized) BUGGY!
        # if self.X_sim_matrix_ is not None and self.y_sim_matrix_ is not None and self.cube_ is not None:
        #     inv_ibc, inv_aic, inv_abi, inv_aii, inv_ibi, inv_iic, inv_iii = CtCoATEnergyComputations._energies_i(
        #         self.X_sim_matrix_, self.y_sim_matrix_, # [..., M, M]
        #         source_sim_vect, outcome_sim_vect, # [..., M]
        #         reflexive_sim_source=source_sim_reflexive, reflexive_sim_outcome=outcome_sim_reflexive, # [...] or []
        #         exclude_impossible=False)
            
        #     # from [n, n, n] to [n, n, n+1]
        #     updated_meatcube.cube_ = torch.cat([self.cube_, inv_abi.squeeze(-3).unsqueeze(-1)], dim=-1)

        #     # from [n, n].[n, 1] to [n, n+1]: add the symmetric component of the vector where the diagonal will be
        #     inv_aic = torch.cat([inv_aic, inv_aii.unsqueeze(-1)], dim=-1)
        #     # from [n, n, n+1].[n, n+1] to [n, n+1, n+1]
        #     updated_meatcube.cube_ = torch.cat([updated_meatcube.cube_, inv_aic.unsqueeze(-2)], dim=-2)

        #     # from [n].[] to [n+1]
        #     inv_iic = torch.cat([inv_iic, inv_iii.unsqueeze(-1)], dim=-1)
        #     # from [n, n].[n, 1] to [n, n+1] to [n+1, n+1]
        #     inv_ibc = torch.cat([inv_ibc, inv_ibi.unsqueeze(-1)], dim=-1)
        #     inv_ibc = torch.cat([inv_ibc, inv_iic.unsqueeze(-2)], dim=-2)
        #     # from [n, n+1, n+1].[n+1, n+1] to [n+1, n+1, n+1]
        #     updated_meatcube.cube_ = torch.cat([updated_meatcube.cube_, inv_ibc.unsqueeze(-3)], dim=-3)

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

        for param in list(self.__dict__.keys()): #[self.X_sim_matrix_, self.y_sim_matrix_, self.y_sim_vectors_, self.cube_]:
            if param.endswith("_") and getattr(self, param) is not None and isinstance(getattr(self, param), torch.Tensor):
                setattr(self, param, getattr(self, param).to(self.device_))

        return self.device_



###########################################################
# TODO: check output shape

    def energy_cb(self, as_tensor=False):
        # self._compute_sim_matrix()
        
        # e = 0
        # for i in range(len(self)):
        #     for (j,k) in combinations(range(len(self)),2):
        #         e += (1 - (self.X_sim_matrix_[i,j]-self.X_sim_matrix_[i,k])*(self.y_sim_matrix_[i,j]-self.y_sim_matrix_[i,k]))

        # e = (e + (len(self)**2)/2)/(len(self)**3)

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
    

    def decrement_scores(self,
            X_ref: Iterable[SourceSpaceElement],
            y_ref: Iterable[OutcomeSpaceElement],
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            as_tensor=False,
            tqdm_verbose = TQDM_VERBOSE,
            **kwargs) -> np.ndarray[float]:
        """Compute the competence of the case base w.r.t a test set, or if an `index` is provided, the contribution of \
        the corresponding case to the competence.

        Lower values correspond to less desirable cases.

        :param index: 
            if provided, computes the contribution (`Cᵢ(CB, ...)`) of the case at `index` (`CBᵢ`) to the competence,
            i.e., the difference of the competence with (`CB`) and without (`CB/CBᵢ`) the case:
            `Cᵢ(CB, ...) = C(CB, ...) - C(CB/CBᵢ, ...)`.
        :param strategy:
            If `strategy` = "MCE", use the minimum classification error loss:
            `ℓ(CB, cₜ) = ℓmce(CB, cₜ) = - ( E(CB ∪ {(sₜ, rₜ)})) - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')})) )`.

            If `strategy` = "hinge", use the hinge competence (hinge, zero_division=0 competence = - hinge loss):
            `ℓ(CB, cₜ) = - max(0, λ + ℓmce(CB, cₜ))`.
        :param margin: If `strategy` = "hinge", `λ=margin`.
        :param aggregation:
            If `aggregation` = None or "none", returns `ℓ(CB, cₜ)` the competence with regard to each test case `cₜ`.
            If `aggregation` = "sum", returns the sum of `ℓ(CB, cₜ)` over all the test cases.
            If `aggregation` = "mean", returns the average of `ℓ(CB, cₜ)` over all the test cases.
        :param normalize: (Deprecated) If True, will normalize the competence by the cube of the CB size.
        :param batch_size:
            If `index` is None, this parameter has no effect.
            If `batch_size` <= 0, the contribution of all cases in `index` are computed at once (high memory impact).
            If `batch_size` = 1, the contribution of each case in `index` is computed one after the other (low memory impact).
            If `batch_size` > 1, the contribution of cases in `index` are computed by batches of `batch_size` indices.
        :return:
            If `aggregation` = None or "none", Size: [|S|] if index is None or int, [|index|, |S|] otherwise
            If `aggregation` = "sum" or "mean", Size: [] if index is None or int, [|index|] otherwise
        """
        # compute necessary data, if need be
        self._compute_sim_matrix()
        self._compute_outcome_sim_vectors()

        # computes the similarity of the new case to the ones in the CB
        X_sim_vectors = self._source_sim_vect(X_ref) # [|S|, |CB|]
        reflexive_sim_X = self._source_sim_reflexive(X_ref) # [|S|]
        reflexive_sim_y = self._outcome_sim_reflexive(y_ref) # [|S|]

        reflexive_sim_y = reflexive_sim_y.unsqueeze(-1) # [|s|, 1]
        reflexive_sim_X = reflexive_sim_X.unsqueeze(-1) # [|s|, 1]

        # inversion_rates: [|S|, |R|]
        inversion_rates = CtCoATEnergyComputations._gamma_i(
            self.X_sim_matrix_.unsqueeze(0).unsqueeze(0), # [1, 1, |CB|, |CB|]
            self.y_sim_matrix_.unsqueeze(0).unsqueeze(0), # [1, 1, |CB|, |CB|]
            X_sim_vectors.unsqueeze(1), # [|S|, 1, |CB|]
            self.y_sim_vectors_.unsqueeze(0), # [1, |R|, |CB|]
            reflexive_sim_source=reflexive_sim_X,
            reflexive_sim_outcome=reflexive_sim_y)
        
        # if we want the contribution of a particular case of the case base,
        # we need to subtract the competence of the case base without said case

        # --------- All the cases at once ---------
        index = list(range(len(self)))
        #try:
        batch_size = len(index)
        inversion_rates_i = self._decrement_scores_batched(
            index,
            reflexive_sim_X,
            reflexive_sim_y,
            X_sim_vectors,
            batch_size=batch_size,
            auto_batch_size=True
        )
        # except torch.cuda.OutOfMemoryError:
        #     batch_size = 1
        #     # batch_size_power = int(np.floor(np.log2(len(index))).item())
        #     # batch_size = 2**batch_size_power
        #     inversion_rates_i = self._decrement_scores_batched(
        #         index,
        #         reflexive_sim_X,
        #         reflexive_sim_y,
        #         X_sim_vectors,
        #         batch_size=batch_size,
        #         auto_batch_size=False
        #     )

        # compare the true outcome with the other outcomes
        true_outcome_index = self._outcome_index(y_ref)
        mask = torch.arange(
            inversion_rates.size(-1),
            device=inversion_rates.device
        ).unsqueeze(0) == true_outcome_index.unsqueeze(1)
        l_mce = inversion_rates[mask] - inversion_rates[~mask].min(dim=-1).values
        mask_max = (mask.unsqueeze(0)) * (inversion_rates_i.max().detach() + 1) # trick to "exclude" the mask from the min
        l_mce_i = inversion_rates_i[:,mask] - (inversion_rates_i + mask_max).min(dim=-1).values
        # l_mce: [|S|]
        # l_mce_i: [|index|, |S|]

        # if hinge loss, modify a bit before aggregation
        if strategy=="hinge":
            l = (margin + l_mce).clamp(min=0)
            l_i = (margin + l_mce_i).clamp(min=0)
        else:
            l = l_mce
            l_i = l_mce_i
        # l: [|S|]
        # l_i: [|index|, |S|]

        # moving from the loss to the competence (and normalizing, if need be)
        l = -l
        l_i = -l_i

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
                                reflexive_sim_X,
                                reflexive_sim_y,
                                X_sim_vectors,
                                batch_size,
                                auto_batch_size=True,
                                keep_on_cuda: Literal[True, False, "auto"] = True,
                                tqdm_verbose = TQDM_VERBOSE,
                                **kwargs) -> np.ndarray[float]:
        backup_device = None
        if auto_batch_size:
            n_cb = len(self) # |CB|
            n_outcomes = self.y_sim_vectors_.size(0) # |R|
            n_ref = X_sim_vectors.size(0) # |S|
            mem_source_sim_matrix_i =   estimate_mem_utilisation(self.X_sim_matrix_)  #dtype=self.X_sim_matrix_.dtype, size=[n_cb-1, n_cb-1])
            mem_outcome_sim_matrix_i =  estimate_mem_utilisation(self.y_sim_matrix_)  #dtype=self.y_sim_matrix_.dtype, size=[n_cb-1, n_cb-1])
            mem_source_sim_vectors_i =  estimate_mem_utilisation(X_sim_vectors)       #dtype=self.X_sim_matrix_.dtype, size=[n_ref, n_cb-1])
            mem_outcome_sim_vectors_i = estimate_mem_utilisation(self.y_sim_vectors_) #dtype=self.y_sim_vectors_.dtype, size=[n_outcomes, n_cb-1])
            mem_inversion_rates_i_ =    estimate_mem_utilisation(dtype=float, size=[n_ref, n_outcomes])
            mem_gamma_i = CtCoATEnergyComputations._estimate_memory_gamma_i(
                self.X_sim_matrix_.unsqueeze(0).unsqueeze(0),
                self.y_sim_matrix_.unsqueeze(0).unsqueeze(0), 
                X_sim_vectors.unsqueeze(1),
                self.y_sim_vectors_.unsqueeze(0))

            # per batch, wrt. CtCoATEnergyComputations._gamma_i()'s formula : [...] = |S|, |R|; [M] = |CB|-1
            # formula = batch * (3 * ([M, M] + [M]) + 7)
            #         = |S| * |R| * (3 * ((|CB|-1)**2 + |CB|-1) + 7)
            #         = |S| * |R| * (3 * ((|CB|-1) * |CB|) + 7)
            # mem_gamma_i = estimate_mem_utilisation(dtype=float, size=[n_ref, n_outcomes, 3, ((n_cb-1)*n_cb-1) + 7])
            # mem_gamma_i *= 1.5 # extra 50% for safety reasons

            operations = [mem_source_sim_matrix_i, mem_outcome_sim_matrix_i, mem_source_sim_vectors_i, mem_outcome_sim_vectors_i]
            mem_batch_input = sum(operations) + max(operations) + mem_gamma_i 
            # "sum + max" for the operations because "sum" counts the space to store the result and "max" counts the extra space to do the operation itself (usually the same amount)
            
            mem_batch_input = mem_source_sim_matrix_i + mem_outcome_sim_matrix_i + mem_source_sim_vectors_i + mem_outcome_sim_vectors_i + mem_gamma_i

            if keep_on_cuda == "auto":
                keep_on_cuda = True
            if keep_on_cuda:
                fixed_overhead = mem_inversion_rates_i_ * len(index)
            else:
                fixed_overhead = mem_inversion_rates_i_
            
            SAFETY_FACTOR = 1.5
            mem_batch_input *= SAFETY_FACTOR
            fixed_overhead *= SAFETY_FACTOR
            
            torch.cuda.empty_cache()
            batch_size = estimate_batch_size(self.device_, mem_batch_input, fixed_overhead=fixed_overhead)
            #logger.info
            
            if batch_size == 0:
                backup_device = self.device()
                self.to_device("cpu")
                reflexive_sim_X = reflexive_sim_X.cpu()
                reflexive_sim_y = reflexive_sim_y.cpu()
                X_sim_vectors = X_sim_vectors.cpu()
                batch_size = 1#len(index)
                #raise torch.cuda.OutOfMemoryError()
                if tqdm_verbose: tqdm.write(f"Auto batch size found not enough space on device, temporarily switching to CPU.\t(considering {mem_batch_input/(1024**2):.2f} MiB per batch and overhead of {fixed_overhead/(1024**2):.2f} MiB, {(fixed_overhead + batch_size*mem_batch_input)/(1024**2):.2f} MiB total)")

            else:
                if tqdm_verbose: tqdm.write(f"Auto batch size: {batch_size}.\t(considering {mem_batch_input/(1024**2):.2f} MiB per batch and overhead of {fixed_overhead/(1024**2):.2f} MiB, {(fixed_overhead + batch_size*mem_batch_input)/(1024**2):.2f} MiB total)")
        

        # try:
        inversion_rates_i = []
        index_batches = [index[i:i+batch_size] for i in range(0, len(index), batch_size)]
        for index_batch in index_batches:
            # self.X_sim_matrix_: # [|CB|, |CB|]
            # self.y_sim_matrix_: # [|CB|, |CB|]
            # X_sim_vectors: [|S|, |CB|]
            # self.y_sim_vectors_: [|R|, |CB|]
            source_sim_matrix_i = torch.stack([
                remove_index(self.X_sim_matrix_, i, dims=[-1,-2])
                for i in index_batch
            ], dim=0).detach() # [|index_batch|, |CB|-1, |CB|-1]
            outcome_sim_matrix_i = torch.stack([
                remove_index(self.y_sim_matrix_, i, dims=[-1,-2])
                for i in index_batch
            ], dim=0).detach() # [|index_batch|, |CB|-1, |CB|-1]
            source_sim_vectors_i = torch.stack([
                remove_index(X_sim_vectors, i, dims=[-1])
                for i in index_batch
            ], dim=0).detach() # [|index_batch|, |S|, |CB|-1]
            outcome_sim_vectors_i = torch.stack([
                remove_index(self.y_sim_vectors_, i, dims=[-1])
                for i in index_batch
            ], dim=0).detach() # [|index_batch|, |R|, |CB|-1]
            # inversion_rates_i_: [|index_batch|, |S|, |R|]
            inversion_rates_i_ = CtCoATEnergyComputations._gamma_i(
                source_sim_matrix_i.unsqueeze(1).unsqueeze(1), # [|index_batch|, 1, 1, |CB|-1, |CB|-1]
                outcome_sim_matrix_i.unsqueeze(1).unsqueeze(1), # [|index_batch|, 1, 1, |CB|-1, |CB|-1]
                source_sim_vectors_i.unsqueeze(2), # [|index_batch|, |S|, 1, |CB|-1]
                outcome_sim_vectors_i.unsqueeze(1), # [|index_batch|, 1, |R|, |CB|-1]
                reflexive_sim_source=reflexive_sim_X.unsqueeze(0),
                reflexive_sim_outcome=reflexive_sim_y.unsqueeze(0)).detach()
            
            if not keep_on_cuda: # remove from GPU if necessary
                inversion_rates_i_ = inversion_rates_i_.cpu()
        
            inversion_rates_i.append(inversion_rates_i_)
            del source_sim_matrix_i, outcome_sim_matrix_i, source_sim_vectors_i, outcome_sim_vectors_i
            torch.cuda.memory.empty_cache()
        
        if len(inversion_rates_i) > 1:
            inversion_rates_i = torch.cat(inversion_rates_i, dim=0)
        else:
            inversion_rates_i = inversion_rates_i[0]

        if backup_device is not None:
            self.to_device(backup_device)
            inversion_rates_i_ = inversion_rates_i_.to(self.device_)
        elif not keep_on_cuda: # put back on GPU if necessary
            inversion_rates_i_ = inversion_rates_i_.to(self.device_)
        
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