from __future__ import annotations # for self-referring type hints
import torch, numpy as np
import pandas as pd
from typing import Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List
from scipy.spatial.distance import squareform, pdist, cdist

try:
    from .torch_backend import MeATCubeEnergyComputations, remove_index, append_symmetric, NORMALIZE
except ImportError as e:
    try:
        from torch_backend import MeATCubeEnergyComputations, remove_index, append_symmetric, NORMALIZE
    except ImportError:
        raise e

SourceSpaceElement = TypeVar('SourceSpaceElement')
OutcomeSpaceElement = TypeVar('OutcomeSpaceElement')
NumberOrBool = Union[float, int, bool]

def numpy_source_or_outcome(values):
    if isinstance(values, torch.Tensor):
        return values.numpy()
    elif isinstance(values, (pd.DataFrame, pd.Series)):
        return values.to_numpy()
    elif isinstance(values, np.ndarray):
        return values
    else:
        try:
            return np.array(values, dtype=float)
        except ValueError:
            return np.array(values)

class MeATCubeCB(Generic[SourceSpaceElement, OutcomeSpaceElement]):
    """Collection of tensors and metrics that automate the computations of several metrics based on the number of 
    inversions, and supports addition and deletion of cases.
    
    Based on MeATCube: (Me)asure of the complexity of a dataset for (A)nalogical (T)ransfer using Boolean (Cube)s, or 
    slices of them."""
    def __init__(self,
                 CB_source: Iterable[SourceSpaceElement], CB_outcome: Iterable[OutcomeSpaceElement],
                 potential_outcomes: List[OutcomeSpaceElement],
                 sim_function_source: Callable[[SourceSpaceElement, SourceSpaceElement], NumberOrBool],
                 sim_function_outcome: Callable[[OutcomeSpaceElement, OutcomeSpaceElement], NumberOrBool]):
        self.CB_source = numpy_source_or_outcome(CB_source)
        self.CB_outcome = numpy_source_or_outcome(CB_outcome)
        self.potential_outcomes = potential_outcomes
        self.sim_function_source = sim_function_source
        self.sim_function_outcome = sim_function_outcome
        self.source_sim_matrix = None # [|CB|, |CB|]
        self.outcome_sim_matrix = None # [|CB|, |CB|]
        self.outcome_sim_vectors = None # [|R|, |CB|], one vector per possible outcome
        self.cube = None # [|CB|, |CB|, |CB|]

    def compute_outcome_sim_vectors(self, force_recompute: bool=False) -> None:
        """Computes the similarity vectors for each possible outcime."""
        if force_recompute or self.outcome_sim_vectors is None:
            potential = np.array(self.potential_outcomes).reshape(-1, self.CB_outcome[0].size)
            self.outcome_sim_vectors = torch.tensor(cdist(
                potential, self.CB_outcome.reshape(-1, self.CB_outcome[0].size), metric=self.sim_function_outcome))

    def is_source_list(self, value: Union[SourceSpaceElement, Iterable[SourceSpaceElement]]) -> bool:
        if isinstance(value, torch.Tensor):
            return value.dim() == self.CB_source.ndim
        elif isinstance(value, np.ndarray) or isinstance(value, (pd.DataFrame, pd.Series)):
            return value.ndim == self.CB_source.ndim
        elif isinstance(value, str):
            return False
        elif isinstance(value, Iterable):
            return (self.CB_source.ndim <= 1) or isinstance(value[0], Iterable)
        else:
            return False
    def is_outcome_list(self, value: Union[OutcomeSpaceElement, Iterable[OutcomeSpaceElement]]) -> bool:
        if isinstance(value, torch.Tensor):
            return value.dim() == self.CB_outcome.ndim
        elif isinstance(value, np.ndarray) or isinstance(value, (pd.DataFrame, pd.Series)):
            return value.ndim == self.CB_outcome.ndim
        elif isinstance(value, str):
            return False
        elif isinstance(value, Iterable):
            return (self.CB_outcome.ndim <= 1) or isinstance(value[0], Iterable)
        else:
            return False

    def outcome_index(self, outcome: Union[OutcomeSpaceElement, Iterable[OutcomeSpaceElement]]) -> torch.LongTensor:
        if isinstance(self.potential_outcomes, np.ndarray):
            index = lambda x: np.where(self.potential_outcomes==x)[0][0]
            if self.is_outcome_list(outcome):
                return torch.tensor([index(o) for o in outcome])
            return torch.tensor(index(outcome))
        else:
            if self.is_outcome_list(outcome):
                return torch.tensor([self.potential_outcomes.index(o) for o in outcome])
            return torch.tensor(self.potential_outcomes.index(outcome))

    def compute_sim_matrix(self, force_recompute: bool=False) -> None:
        """Computes the similarity matrices."""
        if force_recompute or self.source_sim_matrix is None:
            self.source_sim_matrix = torch.tensor(squareform(pdist(self.CB_source.reshape(-1,self.CB_source[0].size), metric=self.sim_function_source)))
            self.source_sim_matrix = self.source_sim_matrix.diagonal_scatter(self.source_sim_reflexive(self.CB_source))
        if force_recompute or self.outcome_sim_matrix is None:
            self.outcome_sim_matrix = torch.tensor(squareform(pdist(self.CB_outcome.reshape(-1,self.CB_outcome[0].size), metric=self.sim_function_outcome)))
            self.outcome_sim_matrix = self.outcome_sim_matrix.diagonal_scatter(self.outcome_sim_reflexive(self.CB_outcome))

    def compute_inversion_cube(self, force_recompute: bool=False) -> None:
        """Computes the inversion cube."""
        if force_recompute or self.cube is None:
            # we need the similarity matrices
            self.compute_sim_matrix(force_recompute=force_recompute)

            # then we compute the cube
            self.cube = MeATCubeEnergyComputations._inversion_cube(self.source_sim_matrix, self.outcome_sim_matrix, return_all=False)

    def source_sim_vect(self, sources: Union[SourceSpaceElement, Iterable[SourceSpaceElement]]) -> torch.Tensor:
        """Computes the similarity between the source and the CB in the source space.
        
        If `sources` is an iterable of size `|S|`, the result is of shape `[|S|, |CB|]`.
        If `sources` is a single value, the result is of shape `[|CB|]`.
        """
        sim_vect = torch.tensor(cdist(
            np.array(sources).reshape(-1,self.CB_source[0].size),
            np.array(self.CB_source).reshape(-1,self.CB_source[0].size),
            metric=self.sim_function_source))
        
        return sim_vect
    def source_sim_reflexive(self, sources: Union[SourceSpaceElement, Iterable[SourceSpaceElement]]) -> torch.Tensor:
        """Computes the similarity between the source and itself.
        
        If `sources` is an iterable of size `|S|`, the result is of shape `[|S|]`.
        If `sources` is a single value, the result is of shape `[]`.
        """
        sources = numpy_source_or_outcome(sources)
        if self.is_source_list(sources):
            sim_reflexive = torch.tensor([self.sim_function_source(source, source) for source in sources])
        else:
            sim_reflexive = torch.tensor(self.sim_function_source(sources, sources))
        
        return sim_reflexive
    
    def outcome_sim_vect(self, outcomes: Union[OutcomeSpaceElement, Iterable[OutcomeSpaceElement]]) -> torch.Tensor:
        """Computes the similarity between the outcome and the CB in the outcome space.
        
        If `outcomes` is an iterable of size `|R|`, the result is of shape `[|R|, |CB|]`.
        If `outcomes` is a single value, the result is of shape `[|CB|]`.
        """
        sim_vect = torch.tensor(cdist(
            np.array(outcomes).reshape(-1,self.CB_outcome[0].size),
            np.array(self.CB_outcome).reshape(-1,self.CB_outcome[0].size),
            metric=self.sim_function_outcome))
        
        return sim_vect
    def outcome_sim_reflexive(self, outcomes: Union[OutcomeSpaceElement, Iterable[OutcomeSpaceElement]]) -> torch.Tensor:
        """Computes the similarity between the source and itself.
        
        If `outcomes` is an iterable of size `|R|`, the result is of shape `[|R|]`.
        If `outcomes` is a single value, the result is of shape `[]`.
        """
        if self.is_outcome_list(outcomes):
            sim_reflexive = torch.tensor([self.sim_function_outcome(outcome, outcome) for outcome in outcomes])
        else:
            sim_reflexive = torch.tensor(self.sim_function_outcome(outcomes, outcomes))
        
        return sim_reflexive
    
    def remove(self, index: int) -> MeATCubeCB:
        """Returns a copy of this MeATCubeCB object where case `index` has been removed.
        
        If initialized, will copy and update the similarity matrices and the cube."""
        updated_meatcube = MeATCubeCB(
            CB_source=np.delete(self.CB_source, index, axis=0),
            CB_outcome=np.delete(self.CB_outcome, index, axis=0),
            potential_outcomes=self.potential_outcomes,
            sim_function_source=self.sim_function_source,
            sim_function_outcome=self.sim_function_outcome)

        # Copy the similarity matrices without the row at index nor the row at index (if already initialized)
        if self.source_sim_matrix is not None:
            updated_meatcube.source_sim_matrix = remove_index(self.source_sim_matrix, index, dims=[-1,-2])
        if self.outcome_sim_matrix is not None:
            updated_meatcube.outcome_sim_matrix = remove_index(self.outcome_sim_matrix, index, dims=[-1,-2])
        if self.cube is not None:
            updated_meatcube.cube = remove_index(self.cube, index, dims=[-1,-2,-3])
        return updated_meatcube

    def add(self, case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement) -> MeATCubeCB:
        """Returns a copy of this MeATCubeCB object where case `index` has been removed.
        
        If initialized, will copy and update the similarity matrices and the cube."""
        updated_meatcube = MeATCubeCB(
            CB_source=np.append(self.CB_source, case_source),
            CB_outcome=np.append(self.CB_outcome, case_outcome),
            potential_outcomes=self.potential_outcomes,
            sim_function_source=self.sim_function_source,
            sim_function_outcome=self.sim_function_outcome)
        # Extend the similarity matrix with the new similarity (if already initialized)
        if self.source_sim_matrix is not None:
            source_sim_vect = self.source_sim_vect(case_source)
            source_sim_reflexive = torch.tensor(self.sim_function_source(case_source, case_source))
            updated_meatcube.source_sim_matrix = append_symmetric(
                self.source_sim_matrix, source_sim_vect, source_sim_reflexive.view(-1))
        if self.outcome_sim_matrix is not None:
            outcome_sim_vect = self.outcome_sim_vect(case_outcome)
            outcome_sim_reflexive = torch.tensor(self.sim_function_outcome(case_outcome, case_outcome))
            updated_meatcube.outcome_sim_matrix = append_symmetric(
                self.outcome_sim_matrix, outcome_sim_vect, outcome_sim_reflexive.view(-1))
        
        # Extend the inversion cube with the new inversions (if already initialized)
        if self.source_sim_matrix is not None and self.outcome_sim_matrix is not None and self.cube is not None:
            inv_ibc, inv_aic, inv_abi, inv_aii, inv_ibi, inv_iic, inv_iii = MeATCubeEnergyComputations._inversions_i(
                self.source_sim_matrix, self.outcome_sim_matrix, # [..., M, M]
                source_sim_vect, outcome_sim_vect, # [..., M]
                reflexive_sim_source=source_sim_reflexive, reflexive_sim_outcome=outcome_sim_reflexive, # [...] or []
                exclude_impossible=False)
            
            # from [n, n, n] to [n, n, n+1]
            updated_meatcube.cube = torch.cat([updated_meatcube.cube, inv_abi], dim=-1)

            # from [n, n].[n, 1] to [n, n+1]: add the symmetric component of the vector where the diagonal will be
            inv_aic = torch.cat([inv_aic, inv_aii.unsqueeze(-1)], dim=-1)
            # from [n, n, n+1].[n, n+1] to [n, n+1, n+1]
            updated_meatcube.cube = torch.cat([updated_meatcube.cube, inv_aic.unsqueeze(-2)], dim=-2)

            # from [n].[] to [n+1]
            inv_iic = torch.cat([inv_iic, inv_iii.unsqueeze(-1)], dim=-1)
            # from [n, n].[n, 1] to [n, n+1] to [n+1, n+1]
            inv_ibc = torch.cat([inv_ibc, inv_ibi.unsqueeze(-1)], dim=-1)
            inv_ibc = torch.cat([inv_ibc, inv_iic.unsqueeze(-2)], dim=-2)
            # from [n, n+1, n+1].[n+1, n+1] to [n+1, n+1, n+1]
            updated_meatcube.cube = torch.cat([updated_meatcube.cube, inv_ibc.unsqueeze(-3)], dim=-3)

        return updated_meatcube

    def energy(self, index: Optional[int]=None, normalize=NORMALIZE):
        """Compute the energy of the case base, or if an `index` is provided, the contribution of the corresponding \
        case to the energy."""
        self.compute_inversion_cube()
        if index is None:
            return MeATCubeEnergyComputations._energy(self.cube, normalize=normalize)
        else:
            return MeATCubeEnergyComputations._cube_gamma_i_included(self.cube, index, normalize=normalize)
        
    def energy_new_case(self, 
                        case_source: Union[SourceSpaceElement, Iterable[SourceSpaceElement]],
                        case_outcome: Union[OutcomeSpaceElement, Iterable[OutcomeSpaceElement]],
                        normalize=NORMALIZE):
        """Compute the contribution of a new case to the energy."""
        self.compute_sim_matrix()
        source_sim_vectors = self.source_sim_vect(case_source)
        reflexive_sim_source = self.source_sim_reflexive(case_source)
        reflexive_sim_outcome = self.outcome_sim_reflexive(case_outcome)


        if self.is_outcome_list(case_outcome):
            outcome_sim_vectors = self.outcome_sim_vect(case_outcome)

        else: # avoid recomputing if it is easy to recover
            self.compute_outcome_sim_vectors()
            outcome_index = self.outcome_index(case_outcome)
            outcome_sim_vectors = self.outcome_sim_vectors.select(-2, outcome_index)

        if outcome_sim_vectors.dim() > 1 and source_sim_vectors.dim() > 1:
            source_sim_vectors = source_sim_vectors.unsqueeze(-2)
            outcome_sim_vectors = outcome_sim_vectors.unsqueeze(-3)
            reflexive_sim_source = reflexive_sim_source.unsqueeze(-1)
            reflexive_sim_outcome = reflexive_sim_outcome.unsqueeze(-2)

        return MeATCubeEnergyComputations._gamma_i(self.source_sim_matrix,
                                 self.outcome_sim_matrix,
                                 source_sim_vectors,
                                 outcome_sim_vectors,
                                 reflexive_sim_source=reflexive_sim_source,
                                 reflexive_sim_outcome=reflexive_sim_outcome,
                                 normalize=normalize)
        
    def predict(self, case_source: Union[SourceSpaceElement, Iterable[SourceSpaceElement]],
                        normalize=NORMALIZE, return_logits=False, return_outcome_indices=False):
        """Compute the contribution of a new case to the energy.
        
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

        inversions = MeATCubeEnergyComputations._gamma_i(self.source_sim_matrix,
                                 self.outcome_sim_matrix,
                                 source_sim_vectors,
                                 outcome_sim_vectors,
                                 reflexive_sim_source=reflexive_sim_source,
                                 reflexive_sim_outcome=reflexive_sim_outcome,
                                 normalize=normalize) # [|S|, |R|] or [|R|]
        
        pred_outcome_index = inversions.argmin(dim=-1)
        if return_outcome_indices:
            pred_outcome = pred_outcome_index
        else:
            pred_outcome = np.vectorize(self.potential_outcomes.__getitem__)(pred_outcome_index.numpy())
        if return_logits:
            return pred_outcome, inversions
        else:
            return pred_outcome
        
        
    def influence(self,
                   test_cases_sources: Iterable[SourceSpaceElement],
                   test_cases_outcomes: Iterable[OutcomeSpaceElement],
                   strategy: Literal["MCE", "hinge"]="hinge",
                   margin: float=0.1,
                   aggregation: Literal[None, "none", "sum", "mean"]="mean",
                   normalize=NORMALIZE) -> torch.Tensor:
        """Contribution of each case in the CB to the global competence.
        
        Short-hand for `self.competence(..., index=list(range(len(self.CB_source))))`."""
        return self.competence(test_cases_sources,
                               test_cases_outcomes,
                               index=list(range(len(self.CB_source))),
                               strategy=strategy,
                               margin=margin,
                               aggregation=aggregation,
                               normalize=normalize)
    
    def competence(self,
                   test_cases_sources: Iterable[SourceSpaceElement],
                   test_cases_outcomes: Iterable[OutcomeSpaceElement],
                   index: Optional[Union[int, List[int]]]=None,
                   strategy: Literal["MCE", "hinge"]="hinge",
                   margin: float=0.1,
                   aggregation: Literal[None, "none", "sum", "mean"]="mean",
                   normalize=NORMALIZE) -> torch.Tensor:
        """Compute the competence of the case base w.r.t a test set, or if an `index` is provided, the contribution of \
        the corresponding case to the competence.

        :param index: 
            if provided, computes the contribution ()`Cᵢ(CB, ...)`) of the case at `index` (`CBᵢ`) to the competence,
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
        :return:
            If `aggregation` = None or "none", Size: [|S|] if index is None or int, [|index|, |S|] otherwise
            If `aggregation` = "sum" or "mean", Size: [] if index is None or int, [|index|] otherwise
        """
        # compute necessary data, if need be
        self.compute_sim_matrix()
        self.compute_outcome_sim_vectors()

        source_sim_vectors = self.source_sim_vect(test_cases_sources) # [|S|, |CB|]

        if self.is_outcome_list(test_cases_outcomes):
            reflexive_sim_outcome = self.outcome_sim_reflexive(test_cases_outcomes).unsqueeze(-1) # [|s|, 1]
        else:
            reflexive_sim_outcome = self.outcome_sim_reflexive(test_cases_outcomes)
        if self.is_source_list(test_cases_sources):
            reflexive_sim_source = self.source_sim_reflexive(test_cases_sources).unsqueeze(-1) # [|s|, 1]
        else:
            reflexive_sim_source = self.source_sim_reflexive(test_cases_sources)

        # inversion_rates: [|S|, |R|]
        inversion_rates = MeATCubeEnergyComputations._gamma_i(
            self.source_sim_matrix.unsqueeze(0).unsqueeze(0), # [1, 1, |CB|, |CB|]
            self.outcome_sim_matrix.unsqueeze(0).unsqueeze(0), # [1, 1, |CB|, |CB|]
            source_sim_vectors.unsqueeze(1), # [|S|, 1, |CB|]
            self.outcome_sim_vectors.unsqueeze(0), # [1, |R|, |CB|]
            reflexive_sim_source=reflexive_sim_source,
            reflexive_sim_outcome=reflexive_sim_outcome,
            normalize=normalize)
        
        # if we want the contribution of a particular case of the case base,
        # we need to subtract the competence of the case base without said case
        if index is not None:
            if isinstance(index, int):
                was_list = False
                index = [index]
            else:
                was_list = True
            source_sim_matrix_i = torch.stack([
                remove_index(self.source_sim_matrix, i, dims=[-1,-2])
                for i in index
            ], dim=0)
            outcome_sim_matrix_i = torch.stack([
                remove_index(self.outcome_sim_matrix, i, dims=[-1,-2])
                for i in index
            ], dim=0)
            source_sim_vectors_i = torch.stack([
                remove_index(source_sim_vectors, i, dims=[-1])
                for i in index
            ], dim=0)
            outcome_sim_vectors_i = torch.stack([
                remove_index(self.outcome_sim_vectors, i, dims=[-1])
                for i in index
            ], dim=0)
            # inversion_rates_i: [|index|, |S|, |R|]
            inversion_rates_i = MeATCubeEnergyComputations._gamma_i(
                source_sim_matrix_i.unsqueeze(1).unsqueeze(1), # [|index|, 1, 1, |CB|-1, |CB|-1]
                outcome_sim_matrix_i.unsqueeze(1).unsqueeze(1), # [|index|, 1, 1, |CB|-1, |CB|-1]
                source_sim_vectors_i.unsqueeze(2), # [|index|, |S|, 1, |CB|-1]
                outcome_sim_vectors_i.unsqueeze(1), # [|index|, 1, |R|, |CB|-1]
                reflexive_sim_source=reflexive_sim_source.unsqueeze(0),
                reflexive_sim_outcome=reflexive_sim_outcome.unsqueeze(0),
                normalize=normalize)
        
        # compare the true outcome with the other outcomes
        true_outcome_index = self.outcome_index(test_cases_outcomes)
        mask = torch.arange(inversion_rates.size(-1), device=inversion_rates.device).unsqueeze(0) == true_outcome_index.unsqueeze(1)
        if len(self.potential_outcomes) == 2:
            l_mce = inversion_rates[~mask] - inversion_rates[mask]
            if index is not None:
                l_mce_i = inversion_rates_i[:,~mask] - inversion_rates_i[:,mask]
        else:
            l_mce = inversion_rates[~mask].min(dim=-1).values - inversion_rates[mask]
            if index is not None:
                mask_max = (mask.unsqueeze(0)) * (inversion_rates_i.max().detach() + 1) # trick to "exclude" the mask from the min
                l_mce_i = (inversion_rates_i + mask_max).min(dim=-1).values - inversion_rates_i[:,mask]
        # l_mce: [|S|]
        # l_mce_i: [|index|, |S|]

        # if hinge loss, modify a bit before aggregation
        if strategy=="hinge":
            l = -(margin - l_mce).clamp(min=0)
            if index is not None:
                l_i = -(margin - l_mce_i).clamp(min=0)
        else:
            l = l_mce
            if index is not None:
                l_i = l_mce_i
        # l: [|S|]
        # l_i: [|index|, |S|]


        if index is not None:
            l = l.unsqueeze(0) - l_i 
            # l: [|index|, |S|]

        # aggregate the results
        if aggregation == "sum":
            l = l.sum(dim = -1, dtype=float)
        elif aggregation == "mean":
            l = l.mean(dim = -1, dtype=float)
        # l: if aggregation is None or "none": [|S|] if index is None or int, [|index|, |S|] if List[int]
        # l: if aggregation is "sum" or "mean": [] if index is None or int, [|index|] if List[int]
        return l if (index is None or was_list) else l.squeeze(0)
