from __future__ import annotations # for self-referring type hints
import torch, numpy as np
import pandas as pd
from typing import Any, Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List
from collections.abc import Sequence
from scipy.spatial.distance import squareform, pdist, cdist
from tqdm.auto import tqdm

from meatcube2.AbstractEnergyBasedPredictor import ACaseBaseEnergyPredictor

try:
    from .meatcube_torch_functional import MeATCubeEnergyComputations, remove_index, append_symmetric, NORMALIZE, pairwise_dist, cart_dist
    from .utils import to_numpy_array
    from .defaults import MEATCUBE_COMPETENCE_NORMALIZE_BY_MAX_COMPETENCE
    from .AbstractEnergyBasedClassifier import ACaseBaseEnergyClassifier, SourceSpaceElement, OutcomeSpaceElement
except ImportError as e:
    try:
        from meatcube_torch_functional import MeATCubeEnergyComputations, remove_index, append_symmetric, NORMALIZE, pairwise_dist, cart_dist
        from utils import to_numpy_array
        from defaults import MEATCUBE_COMPETENCE_NORMALIZE_BY_MAX_COMPETENCE
        from AbstractEnergyBasedClassifier import ACaseBaseEnergyClassifier, SourceSpaceElement, OutcomeSpaceElement
    except ImportError:
        raise e

NumberOrBool = Union[float, int, bool]

class MeATCubeCB(ACaseBaseEnergyClassifier):
    """Collection of tensors and metrics that automate the computations of several metrics based on the number of 
    inversions, and supports addition and deletion of cases.
    
    Based on MeATCube: (Me)asure of the complexity of a dataset for (A)nalogical (T)ransfer using Boolean (Cube)s, or 
    slices of them."""
    _X = None
    _y = None
    potential_outcomes_ = None
    source_sim_matrix_ = None # [|CB|, |CB|]
    outcome_sim_matrix_ = None # [|CB|, |CB|]
    outcome_sim_vectors_ = None # [|R|, |CB|], one vector per possible outcome
    cube_ = None

    def __init__(self, sim_X: Callable[[Any, Any], float], sim_y: Callable[[Any, Any], float]):
        super().__init__(sim_X, sim_y)
        

    def __getitem__(self, index) -> Tuple:
        return self._X[index], self._y[index]
    def __len__(self) -> int:
        return max(len(self._X), len(self._y))
    def decrement_scores(self, X_ref, y_ref, **competence_kwargs) -> List[float]:
        pass#raise NotImplementedError
    def increment_scores(self, X_candidate, y_candidate, X_ref, y_ref, **competence_kwargs) -> List[float]:
        pass#raise NotImplementedError
        
    def fit_input(self,
                 X: Iterable[SourceSpaceElement], y: Iterable[OutcomeSpaceElement],
                 potential_outcomes: List[OutcomeSpaceElement]="auto",
                 device = "auto") :
        """Initializes attributes depending on the CB."""
        self._X = to_numpy_array(X)
        self._y = to_numpy_array(y)
        if potential_outcomes == "auto":
            self.potential_outcomes_ = np.unique(self._y)
        else:
            self.potential_outcomes_ = potential_outcomes
        self.source_sim_matrix_ = None # [|CB|, |CB|]
        self.outcome_sim_matrix_ = None # [|CB|, |CB|]
        self.outcome_sim_vectors_ = None # [|R|, |CB|], one vector per possible outcome
        self.cube_ = None # [|CB|, |CB|, |CB|]
        if device == "auto":
            if torch.cuda.is_available():
                self.device_ = torch.device("cuda")
            else:
                self.device_ = torch.device("cpu")
        else:
            self.device_ = device
        super(Sequence).__init__()

    def remove(self, index: int) -> MeATCubeCB:
        """Returns a copy of this MeATCubeCB object where case `index` has been removed.
        
        If initialized, will copy and update the similarity matrices and the cube."""
        if isinstance(index, torch.Tensor):
            index_cpu=int(index.cpu().item())
        else:
            index_cpu=index
        updated_meatcube = MeATCubeCB(sim_X=self.sim_X,sim_y=self.sim_y)
        updated_meatcube.fit_input(
            X=np.delete(self._X, index_cpu, axis=0),
            y=np.delete(self._y, index_cpu, axis=0),
            potential_outcomes=self.potential_outcomes_,
            device=self.device_)

        # Copy the similarity matrices without the row at index nor the row at index (if already initialized)
        if self.source_sim_matrix_ is not None:
            updated_meatcube.source_sim_matrix_ = remove_index(self.source_sim_matrix_, index, dims=[-1,-2])
        if self.outcome_sim_matrix_ is not None:
            updated_meatcube.outcome_sim_matrix_ = remove_index(self.outcome_sim_matrix_, index, dims=[-1,-2])
        if self.cube_ is not None:
            updated_meatcube.cube_ = remove_index(self.cube_, index, dims=[-1,-2,-3])
        
        return updated_meatcube
    
    def add(self, case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement) -> MeATCubeCB:
        """Returns a copy of this MeATCubeCB object where case `index` has been removed.
        
        If initialized, will copy and update the similarity matrices and the cube."""
        updated_meatcube = MeATCubeCB(sim_X=self.sim_X,sim_y=self.sim_y)
        updated_meatcube.fit_input(
            X=np.append(self._X, case_source),
            y=np.append(self._y, case_outcome),
            potential_outcomes=self.potential_outcomes_,
            device=self.device_)
        
        # Extend the similarity matrix with the new similarity (if already initialized)
        if self.source_sim_matrix_ is not None:
            source_sim_vect = self._source_sim_vect(case_source)
            source_sim_reflexive = torch.tensor(self._sim_function_source(case_source, case_source))
            updated_meatcube.source_sim_matrix_ = append_symmetric(
                self.source_sim_matrix_, source_sim_vect, source_sim_reflexive.view(-1))
        if self.outcome_sim_matrix_ is not None:
            outcome_sim_vect = self._outcome_sim_vect(case_outcome)
            outcome_sim_reflexive = torch.tensor(self.sim_y(case_outcome, case_outcome))
            updated_meatcube.outcome_sim_matrix_ = append_symmetric(
                self.outcome_sim_matrix_, outcome_sim_vect, outcome_sim_reflexive.view(-1))
        
        # Extend the inversion cube with the new inversions (if already initialized)
        if self.source_sim_matrix_ is not None and self.outcome_sim_matrix_ is not None and self.cube_ is not None:
            inv_ibc, inv_aic, inv_abi, inv_aii, inv_ibi, inv_iic, inv_iii = MeATCubeEnergyComputations._inversions_i(
                self.source_sim_matrix_, self.outcome_sim_matrix_, # [..., M, M]
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

        return updated_meatcube
    
    def energy_cb(self, normalize=NORMALIZE):
        """Compute the energy of the case base, or if an `index` is provided, the contribution of the corresponding \
        case to the energy."""
        self._compute_inversion_cube()
        return MeATCubeEnergyComputations._energy(self.cube_, normalize=normalize)
    def energy_case_from_cb(self, index: int, normalize=NORMALIZE):
        """Compute the energy of the case base, or if an `index` is provided, the contribution of the corresponding \
        case to the energy."""
        self._compute_inversion_cube()
        return MeATCubeEnergyComputations._cube_gamma_i_included(self.cube_, index, normalize=normalize)
    def energy_case_new(self, case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement, normalize=NORMALIZE) -> float:
        """Compute the contribution of a new case to the energy."""
        self._compute_sim_matrix()
        source_sim_vectors = self._source_sim_vect(case_source)
        reflexive_sim_source = self._source_sim_reflexive(case_source)
        reflexive_sim_outcome = self._outcome_sim_reflexive(case_outcome)


        if self._is_outcome_list(case_outcome):
            outcome_sim_vectors = self._outcome_sim_vect(case_outcome)

        else: # avoid recomputing if it is easy to recover
            self._compute_outcome_sim_vectors()
            outcome_index = self.outcome_index(case_outcome)
            outcome_sim_vectors = self.outcome_sim_vectors_.select(-2, outcome_index)

        if outcome_sim_vectors.dim() > 1 and source_sim_vectors.dim() > 1:
            source_sim_vectors = source_sim_vectors.unsqueeze(-2)
            outcome_sim_vectors = outcome_sim_vectors.unsqueeze(-3)
            reflexive_sim_source = reflexive_sim_source.unsqueeze(-1)
            reflexive_sim_outcome = reflexive_sim_outcome.unsqueeze(-2)

        return MeATCubeEnergyComputations._gamma_i(self.source_sim_matrix_,
                                 self.outcome_sim_matrix_,
                                 source_sim_vectors,
                                 outcome_sim_vectors,
                                 reflexive_sim_source=reflexive_sim_source,
                                 reflexive_sim_outcome=reflexive_sim_outcome,
                                 normalize=normalize)
    
    def predict_one(self,
                    case_source: SourceSpaceElement, 
                    candidate_outcomes=None, 
                    return_outcome_id=False, 
                    return_outcome_energy=False, 
                    normalize=NORMALIZE,
                    **kwargs) -> OutcomeSpaceElement | Tuple[OutcomeSpaceElement | int] | Tuple[OutcomeSpaceElement | float] | Tuple[OutcomeSpaceElement | int | float]:
        
        # TODO integrate candidate_outcomes

        # prepare the data as necessary
        self._compute_sim_matrix()
        self._compute_outcome_sim_vectors()

        source_sim_vectors = self._source_sim_vect(case_source) # [|S|] (or [1] if case_source is not a list of sources)
        reflexive_sim_source = self._source_sim_reflexive(case_source) # [|S|] (or [1] if case_source is not a lis of sources)
        reflexive_sim_outcome = self._outcome_sim_reflexive(self.potential_outcomes_) # [|R|]
        outcome_sim_vectors = self.outcome_sim_vectors_ # [|R|, |CB|]
        
        inversions = MeATCubeEnergyComputations._gamma_i(self.source_sim_matrix_,
                                 self.outcome_sim_matrix_,
                                 source_sim_vectors,
                                 outcome_sim_vectors,
                                 reflexive_sim_source=reflexive_sim_source,
                                 reflexive_sim_outcome=reflexive_sim_outcome,
                                 normalize=normalize) # [|S|, |R|] or [|R|]

        # get the minimum energy outcome
        pred_outcome_index = int(inversions.argmin(dim=-1).cpu().item())
        pred_outcome = np.vectorize(self.potential_outcomes_.__getitem__)([pred_outcome_index])
        if return_outcome_energy:
            inversions = inversions.cpu().item()

        if return_outcome_id and return_outcome_energy:
            return pred_outcome, pred_outcome_index, inversions
        elif return_outcome_id:
            return pred_outcome, pred_outcome_index
        elif return_outcome_energy:
            return pred_outcome, inversions
        else:
            return pred_outcome
        
    def predict_multiple(self, 
                         cases_sources: Iterable[SourceSpaceElement],
                         candidate_outcomes=None, 
                         return_outcome_id=False, 
                         return_outcome_energy=False, 
                         normalize=NORMALIZE,
                         **kwargs) -> Iterable | Tuple[Iterable | Iterable[int]] | Tuple[Iterable | Iterable[float]] | Tuple[Iterable | Iterable[int] | Iterable[float]]:
        # prepare the data as necessary
        self._compute_sim_matrix()
        self._compute_outcome_sim_vectors()

        source_sim_vectors = self._source_sim_vect(cases_sources) # [|S|] (or [1] if case_source is not a list of sources)
        reflexive_sim_source = self._source_sim_reflexive(cases_sources) # [|S|] (or [1] if case_source is not a lis of sources)
        reflexive_sim_outcome = self._outcome_sim_reflexive(self.potential_outcomes_) # [|R|]
        outcome_sim_vectors = self.outcome_sim_vectors_ # [|R|, |CB|]
        
        # prepare for new dimensions
        source_sim_vectors = source_sim_vectors.unsqueeze(-2) # [|S|, 1, |CB|]
        outcome_sim_vectors = outcome_sim_vectors.unsqueeze(-3) # [1, |R|, |CB|]
        reflexive_sim_source = reflexive_sim_source.unsqueeze(-1) # [|S|, 1]
        reflexive_sim_outcome = reflexive_sim_outcome.unsqueeze(-2) # [1, |R|]

        inversions = MeATCubeEnergyComputations._gamma_i(self.source_sim_matrix_,
                                 self.outcome_sim_matrix_,
                                 source_sim_vectors,
                                 outcome_sim_vectors,
                                 reflexive_sim_source=reflexive_sim_source,
                                 reflexive_sim_outcome=reflexive_sim_outcome,
                                 normalize=normalize) # [|S|, |R|] or [|R|]

        pred_outcome_index = inversions.argmin(dim=-1).cpu().numpy()
        pred_outcome = np.vectorize(self.potential_outcomes_.__getitem__)(pred_outcome_index)
        if return_outcome_energy:
            inversions = inversions.cpu().numpy()

        if return_outcome_id and return_outcome_energy:
            return pred_outcome, pred_outcome_index, inversions
        elif return_outcome_id:
            return pred_outcome, pred_outcome_index
        elif return_outcome_energy:
            return pred_outcome, inversions
        else:
            return pred_outcome
        
    def _predict(self, case_source: Union[SourceSpaceElement, Iterable[SourceSpaceElement]],
                        normalize=NORMALIZE, return_logits=False, return_outcome_indices=False):
        """OLD .predict method
        
        Compute the contribution of a new case to the energy.
        
        :param return_outcome_indices:
            if False, returns a np.ndarray of outcomes taken from self.potential_outcomes;
            if True, returns an index tensor.
        """
        self.compute_sim_matrix()
        self.compute_outcome_sim_vectors()

        source_sim_vectors = self.source_sim_vect(case_source) # [|S|] (or [1] if case_source is not a lis of sources)
        reflexive_sim_source = self.source_sim_reflexive(case_source) # [|S|] (or [1] if case_source is not a lis of sources)
        reflexive_sim_outcome = self.outcome_sim_reflexive(self.potential_outcomes) # [|R|]
        outcome_sim_vectors = self.outcome_sim_vectors # [|R|, |CB|]
        if self.is_source_list(case_source):
            source_sim_vectors = source_sim_vectors.unsqueeze(-2) # [|S|, 1, |CB|]
            outcome_sim_vectors = outcome_sim_vectors.unsqueeze(-3) # [1, |R|, |CB|]
            reflexive_sim_source = reflexive_sim_source.unsqueeze(-1) # [|S|, 1]
            reflexive_sim_outcome = reflexive_sim_outcome.unsqueeze(-2) # [1, |R|]

        inversions = MeATCubeEnergyComputations._gamma_i(self.source_sim_matrix_,
                                 self.outcome_sim_matrix_,
                                 source_sim_vectors,
                                 outcome_sim_vectors,
                                 reflexive_sim_source=reflexive_sim_source,
                                 reflexive_sim_outcome=reflexive_sim_outcome,
                                 normalize=normalize) # [|S|, |R|] or [|R|]
        
        pred_outcome_index = inversions.argmin(dim=-1)
        if return_outcome_indices:
            pred_outcome = pred_outcome_index
        else:
            pred_outcome = np.vectorize(self.potential_outcomes.__getitem__)(pred_outcome_index.cpu().numpy())
        if return_logits:
            return pred_outcome, inversions
        else:
            return pred_outcome
        




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
        if isinstance(self.potential_outcomes_, np.ndarray):
            index = lambda x: np.where(self.potential_outcomes_==x)[0][0]
            if self._is_outcome_list(outcome):
                return torch.tensor([index(o) for o in outcome]).to(self.device_)
            return torch.tensor(index(outcome)).to(self.device_)
        else:
            if self._is_outcome_list(outcome):
                return torch.tensor([self.potential_outcomes_.index(o) for o in outcome]).to(self.device_)
            return torch.tensor(self.potential_outcomes_.index(outcome)).to(self.device_)

    def _compute_outcome_sim_vectors(self, force_recompute: bool=False) -> None:
        """Computes the similarity vectors for each possible outcime."""
        if force_recompute or self.outcome_sim_vectors_ is None:
            potential = self._prep_outcome_for_dist(self.potential_outcomes_)
            self.outcome_sim_vectors_ = torch.tensor(cdist(
                potential, self._prep_outcome_for_dist(self._y), metric=self.sim_y))
            self.outcome_sim_vectors_ = self.outcome_sim_vectors_.to(self.device_)

    def _compute_sim_matrix(self, force_recompute: bool=False) -> None:
        """Computes the similarity matrices."""
        if force_recompute or self.source_sim_matrix_ is None:
            self.source_sim_matrix_ = torch.tensor(squareform(pdist(self._prep_source_for_dist(self._X), metric=self.sim_X)))
            self.source_sim_matrix_ = self.source_sim_matrix_.diagonal_scatter(self._source_sim_reflexive(self._X))
            self.source_sim_matrix_ = self.source_sim_matrix_.to(self.device_)
        if force_recompute or self.outcome_sim_matrix_ is None:
            self.outcome_sim_matrix_ = torch.tensor(squareform(pdist(self._prep_outcome_for_dist(self._y), metric=self.sim_y)))
            self.outcome_sim_matrix_ = self.outcome_sim_matrix_.diagonal_scatter(self._outcome_sim_reflexive(self._y))
            self.outcome_sim_matrix_ = self.outcome_sim_matrix_.to(self.device_)

    def _compute_inversion_cube(self, force_recompute: bool=False) -> None:
        """Computes the inversion cube."""
        if force_recompute or self.cube_ is None:
            # we need the similarity matrices
            self._compute_sim_matrix(force_recompute=force_recompute)

            # then we compute the cube
            self.cube_ = MeATCubeEnergyComputations._inversion_cube(self.source_sim_matrix_, self.outcome_sim_matrix_, return_all=False)

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