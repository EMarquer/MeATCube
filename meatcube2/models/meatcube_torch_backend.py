import torch
import numpy as np
import pandas as pd
from typing import Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List
from scipy.spatial.distance import squareform, pdist, cdist
from tqdm.auto import tqdm
from ..torch_utils import pop_index
from ..utils import estimate_mem_utilisation

NORMALIZE = False

class MeATCubeEnergyComputations(object):
    """(Me)asure of the complexity of a dataset for (A)nalogical (T)ransfer using Boolean (Cube)s, or slices of them."""
    @staticmethod
    def _energy(cube):
        """The energy of the case base is the """
        e = cube.sum(dim=[-3,-2,-1])
        return e

    @staticmethod
    def _inversion_cube(source_sim: torch.Tensor, outcome_sim: torch.Tensor, source_cube: Optional[torch.Tensor]=None,
                   outcome_cube: Optional[torch.Tensor]=None, return_all=False) -> Union[
                       torch.BoolTensor, Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]]:
        """Computation of the cube of inversions for the Γ indicator.
        
        :param source_sim:  Tensor with coordinates `[a, b]` for `σs(a,b)`.
        :param outcome_sim: Tensor with coordinates `[a, b]` for `σr(a,b)`.
        :param return_all: If true, instead of returning only the inversion cube, returns the comparison cube for the 
        source and the outcome similarities.
        :return: Boolean tensor with coordinates `[a,b,c]` for `σs(a,b) ≥ σs(a,c) ∧ σr(a,b) < σr(a,c)`.

        -------
        Memory usage
        -------
        2 * _cubify (source_cube, outcome_cube) + boolean cube 
            
        i.e., batch_dims * 2 * [M, M, M] * sim_matrix.dtype + batch_dims * [M, M, M] * sim_matrix.bool
        """
        #n = source_sim.size(0)
        #self.s_cube = source_sim.view((n, n, 1)) >= source_sim.view((n, 1, n))
        #self.o_cube = outcome_sim.view((n, n, 1)) < outcome_sim.view((n, 1, n))

        # compute only if necessary
        if source_cube is None: source_cube = MeATCubeEnergyComputations._cubify(source_sim, "source")
        if outcome_cube is None: outcome_cube = MeATCubeEnergyComputations._cubify(outcome_sim, "outcome")

        if source_cube.dim() == 3 & outcome_cube.dim() == 4: source_cube.unsqueeze(0)
        elif source_cube.dim() == 4 & outcome_cube.dim() == 3: outcome_cube.unsqueeze(0)
        cube = source_cube & outcome_cube

        if return_all:
            return cube, source_cube, outcome_cube
        else:
            return cube

    @staticmethod
    def _cubify(sim_matrix: torch.Tensor, comparator: Literal["source","outcome"]="source") -> torch.BoolTensor:
        """Transform the similarity matrix into a cube (or an array of cubes), based on a provided comparator.

        If `sim_matrix` is of size `[M, M]`, `M` the number of cases, it should use the coordinates `[a, b]` for the 
        similarity `σ(a, b)`. Then, the output will be of size `[M, M, M]` with coordinates `[a, b, c]` for:
        - `σ(a, b) >= σ(a, c)` if comparator is 'source';
        - `σ(a, b) < σ(a, c)` if comparator is 'outcome'.

        If `sim_matrix` is of size `[O, M, M]`, `M` the number of cases and `O` the number of outcomes, it should use 
        the coordinates `[o, a, b]` for the similarity :math:`σ(a, b)` given the outcome `o`. Then, the output will be
        of size `[M, M, M]` with coordinates `[o, a, b, c]` for:
        - `σ(a, b) >= σ(a, c)` given the outcome `o` if comparator is 'source';
        - `σ(a, b) < σ(a, c)` given the outcome `o` if comparator is 'outcome'.

        Generalizes for shape `[..., M, M]` with `...` any number of dimensions.

        :param sim_matrix: Tensor of the similarity matrix or stack of similarity matrices.
        :param comparator: Either 'source' or 'outcome'.
        :return: Tensor containing the cube (or an array of cubes) of boolean values.

        -------
        Memory usage
        -------
        batch_dims * [M, M, M] * sim_matrix.dtype
        """
        comparator = (lambda ab, ac: ab >= ac) if comparator=="source" else (lambda ab, ac: ab < ac)
        if sim_matrix.dim() >= 2 and sim_matrix.size(-1) == sim_matrix.size(-1):
            # sim_matrix: [M, M]
            # cube: [M, M, M], coordinates [a, b, c], `a` the anchor
            # works by reshaping sim_matrix as follows
            # comparator([dim anchor, dim a, . ], [dim anchor, . , dim b])
            
            #cube = comparator(sim_matrix.view((m, m, 1)), sim_matrix.view((m, 1, m)))
            #cube = comparator(sim_matrix.view((o, m, m, 1)), sim_matrix.view((o, m, 1, m)))
            cube = comparator(sim_matrix.unsqueeze(-1), sim_matrix.unsqueeze(-2))
        else:
            raise ValueError(f"Unsupported cubification of a matrix of size {sim_matrix.size()}: only works with at" 
                             "least 2 dimensions, with the last two of equal size.")
        
        return cube
    
    @staticmethod
    def _estimate_memory_inversions_i(sim_source, sim_outcome, new_sim_source, new_sim_outcome,
                      reflexive_sim_source=1, reflexive_sim_outcome=1, exclude_impossible=True) -> int:
        """
        -------
        Memory usage
        -------
        batch_dims * (6 * [M, M] + 6 * [M]) * bool
        OR
        batch_dims * (6 * [M, M] + 4 * [M]) * bool
        """
        # get max broadcast sizes
        batch_sizes = list(sim_source.size()[:-2])
        other_inputs = [sim_outcome.size()[:-2], new_sim_source.size()[:-1], new_sim_outcome.size()[:-1]]
        if isinstance(reflexive_sim_source, torch.Tensor): other_inputs += [reflexive_sim_source.size()]
        if isinstance(reflexive_sim_outcome, torch.Tensor): other_inputs += [reflexive_sim_outcome.size()]
        for other_batch_sizes in other_inputs:
            for i in range(len(batch_sizes)):
                batch_sizes[i] = max(batch_sizes[i], other_batch_sizes[i])

        M = sim_source.size(-1)
        mem_estimate_inv_ibc = mem_estimate_inv_aic = mem_estimate_inv_abi = estimate_mem_utilisation(dtype=bool, size=list(batch_sizes) + [M, M])
        mem_estimate_inv_iic = mem_estimate_inv_ibi = estimate_mem_utilisation(dtype=bool, size=list(batch_sizes) + [M])
        if exclude_impossible:
            mem_estimate_inv_aii = 0
            mem_estimate_inv_iii = 0
        else:
            mem_estimate_inv_aii = mem_estimate_inv_iic
            mem_estimate_inv_iii = mem_estimate_inv_iic/M
        
        operations = [mem_estimate_inv_ibc, mem_estimate_inv_aic, mem_estimate_inv_abi, mem_estimate_inv_iic, mem_estimate_inv_ibi, mem_estimate_inv_aii, mem_estimate_inv_iii]
        mem_estimate = sum(operations) + max(operations)
        return mem_estimate

    @staticmethod
    def _inversions_i(sim_source, sim_outcome, new_sim_source, new_sim_outcome,
                      reflexive_sim_source=1, reflexive_sim_outcome=1, exclude_impossible=True) -> (
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):

        """The idea is to compute the inversions for every possible outcome, and make the choice of the right one only afterwards.
        
        For a base `CB` of `M` cases and a new case `i`, computes the inversions involving `i`.
        Generalizes for any dimensions `...`, that can range from 0 dimensions (for the basic Γi) to any number of 
        dimensions for parallel computations. Sizes of the dimensions `...` may differ between the inputs, but most be 
        broadcast-able.

        :param sim_source: Source similarity matrix `σs` in the case base. 
        Size: `[..., M, M]`. Coordinates: `σs[a,b] = σs(a,b) ∀a,b ∈ CB`.
        :param sim_outcome: Outcome similarity matrix `σr` in the case base. 
        Size: `[..., M, M]`. Coordinates: `σr[a,b] = σr(a,b) ∀a,b ∈ CB`.
        :param new_sim_source: Source similarity vector `σs[i]` between the new case `i` and the case base. 
        Size: `[..., M]`. Coordinates: `σs[a] = σs(i,a) ∀a ∈ CB`.
        :param new_sim_outcomes: Outcome similarity vector `σr[i]` between the new case `i` and the case base. 
        Size: `[..., M]`. Coordinates: `σr[a] = σr(i,a) ∀a ∈ CB`.
        :param reflexive_sim_source: Reflexive source similarity `σs(i,i)`.
        Size: `[...]` or constant value.
        :param reflexive_sim_outcome: Reflexive outcome similarity `σr(i,i)`.
        Size: `[...]` or constant value.

        :return: In order:
        - gamma_ibc, gamma_aic, gamma_abi: Size: `[..., M, M]`
        - gamma_aii, gamma_ibi, gamma_iic: Size: `[..., M]`
        - gamma_iii: Size: `[...]`
        """
        # rename in a short manner
        s=sim_source  # [..., M, M]
        o=sim_outcome  # [..., M, M]
        si=new_sim_source  # [..., M]
        oi=new_sim_outcome  # [..., M]
        sii=reflexive_sim_source  # [...] or []
        oii=reflexive_sim_outcome  # [...] or []
        # gamma_ibc: [..., M, 1] . [..., 1, M] -> [..., M, M]
        inv_ibc = ((si.unsqueeze(-1) >= si.unsqueeze(-2)) &
                     (oi.unsqueeze(-1) < oi.unsqueeze(-2))) ## 843417600 bytes
        # gamma_aic: [..., M, 1] . [..., M, M] -> [..., M, M]
        inv_aic = ((si.unsqueeze(-1) >= s) &
                     (oi.unsqueeze(-1) < o)) ## 843417600 bytes
        # gamma_abi: [..., M, M] . [..., M, 1] -> [..., M, M]
        inv_abi = ((s >= si.unsqueeze(-1)) &
                     (o < oi.unsqueeze(-1))) ## 843417600 bytes
        
        if isinstance(sii, torch.Tensor) and sii.dim() == si.dim() - 1:
            sii = sii.unsqueeze(-1)
        if isinstance(oii, torch.Tensor) and oii.dim() == oi.dim() - 1:
            oii = oii.unsqueeze(-1)
        # gamma_ibi: [..., M] . [] -> [..., M]
        inv_ibi = ((si >= sii) &
                    (oi < oii))
        # gamma_iic: [] . [..., M] -> [..., M]
        inv_iic = ((sii >= si) & 
                    (oii < oi))
        
        if exclude_impossible:
            inv_aii = 0 # cannot invert itself
            inv_iii = 0 # cannot invert itself, special case of gamma_aii
        else:
            inv_aii = torch.zeros_like(inv_iic, dtype=int)
            inv_iii = torch.zeros_like(inv_iic.select(-1, 0), dtype=int)

        return inv_ibc, inv_aic, inv_abi, inv_aii, inv_ibi, inv_iic, inv_iii

    @staticmethod
    def _estimate_memory_gamma_i(sim_source, sim_outcome, new_sim_source, new_sim_outcome,
                      reflexive_sim_source=1, reflexive_sim_outcome=1) -> int:
        """
        -------
        Memory usage
        -------
        _inversions_i + (2*5 * batch_dims) * int

        batch_dims * (6 * [M, M] + 4 * [M]) * bool + (10 * batch_dims) * int
        """
        # get max broadcast sizes
        batch_sizes = list(sim_source.size()[:-2])
        other_inputs = [sim_outcome.size()[:-2], new_sim_source.size()[:-1], new_sim_outcome.size()[:-1]]
        if isinstance(reflexive_sim_source, torch.Tensor): other_inputs += [reflexive_sim_source.size()]
        if isinstance(reflexive_sim_outcome, torch.Tensor): other_inputs += [reflexive_sim_outcome.size()]
        for other_batch_sizes in other_inputs:
            for i in range(len(batch_sizes)):
                batch_sizes[i] = max(batch_sizes[i], other_batch_sizes[i])
        
        memory_inversions_i = MeATCubeEnergyComputations._estimate_memory_inversions_i(
            sim_source=sim_source,
            sim_outcome=sim_outcome,
            new_sim_source=new_sim_source,
            new_sim_outcome=new_sim_outcome,
            reflexive_sim_source=reflexive_sim_source,
            reflexive_sim_outcome=reflexive_sim_outcome)
        
        mem_estimate_gamma_ibc = mem_estimate_gamma_aic = mem_estimate_gamma_abi = estimate_mem_utilisation(dtype=int, size=other_batch_sizes)
        mem_estimate_gamma_ibi = mem_estimate_gamma_iic = estimate_mem_utilisation(dtype=int, size=other_batch_sizes)

        operations = [mem_estimate_gamma_ibc, mem_estimate_gamma_aic, mem_estimate_gamma_abi, mem_estimate_gamma_ibi, mem_estimate_gamma_iic]
        mem_estimate = memory_inversions_i + sum(operations) + max(operations)
        return mem_estimate
    
    @staticmethod
    def _gamma_i(sim_source: torch.Tensor, 
                 sim_outcome: torch.Tensor, 
                 new_sim_source: torch.Tensor,
                 new_sim_outcome: torch.Tensor,
                 reflexive_sim_source: Union[float, torch.Tensor]=1, 
                 reflexive_sim_outcome: Union[float, torch.Tensor]=1) -> torch.Tensor:
        """The idea is to compute the inversions for every possible outcome, and make the choice of the right one only afterwards.
        
        For a base `CB` of `M` cases and a new case `i`, computes the inversions involving `i`.
        Generalizes for any dimensions `...`, that can range from 0 dimensions (for the basic Γi) to any number of 
        dimensions for parallel computations. Sizes of the dimensions `...` may differ between the inputs, but most be 
        broadcast-able.

        :param sim_source: Source similarity matrix `σs` in the case base. 
        Size: `[..., M, M]`. Coordinates: `σs[a,b] = σs(a,b) ∀a,b ∈ CB`.
        :param sim_outcome: Outcome similarity matrix `σr` in the case base. 
        Size: `[..., M, M]`. Coordinates: `σr[a,b] = σr(a,b) ∀a,b ∈ CB`.
        :param new_sim_source: Source similarity vector `σs[i]` between the new case `i` and the case base. 
        Size: `[..., M]`. Coordinates: `σs[a] = σs(i,a) ∀a ∈ CB`.
        :param new_sim_outcomes: Outcome similarity vector `σr[i]` between the new case `i` and the case base. 
        Size: `[..., M]`. Coordinates: `σr[a] = σr(i,a) ∀a ∈ CB`.
        :param reflexive_sim_source: Reflexive source similarity `σs(i,i)`.
        Size: `[...]` or constant value.
        :param reflexive_sim_outcome: Reflexive outcome similarity `σr(i,i)`.
        Size: `[...]` or constant value.
        :param normalize: (Deprecated) If True, will normalize the competence by the cube of the CB size.

        :return: The competence of `CB` w.r.t the case `i`.
        """
        inv_ibc, inv_aic, inv_abi, inv_aii, inv_ibi, inv_iic, inv_iii = MeATCubeEnergyComputations._inversions_i(
            sim_source, sim_outcome, # [..., M, M]
            new_sim_source, new_sim_outcome, # [..., M]
            reflexive_sim_source=reflexive_sim_source, reflexive_sim_outcome=reflexive_sim_outcome, # [...] or []
            exclude_impossible=True) ##843417600 bytes 

        # gamma_ibc, gamma_aic, gamma_abi: [..., M, M] -> [...]
        gamma_ibc = inv_ibc.sum(dim=[-2,-1], dtype=int) 
        gamma_aic = inv_aic.sum(dim=[-2,-1], dtype=int) 
        gamma_abi = inv_abi.sum(dim=[-2,-1], dtype=int) 
        
        # gamma_ibi, gamma_iic: [..., M] -> [...]
        gamma_ibi = inv_ibi.sum(dim=-1, dtype=int)
        gamma_iic = inv_iic.sum(dim=-1, dtype=int)
        
        gamma_aii = 0 # cannot invert itself
        gamma_iii = 0 # cannot invert itself, special case of gamma_aii

        return gamma_ibc + gamma_aic + gamma_abi + gamma_aii + gamma_ibi + gamma_iic + gamma_iii
    
    @staticmethod
    def _gamma_i_included(sim_source, sim_outcome, i) -> torch.Tensor:
        """The idea is to compute the inversions for every possible outcome, and make the choice of the right one only afterwards.
        
        For a base `CB` of `M` cases and a new case `i`, computes the inversions involving `i`.
        Generalizes for any dimensions `...`, that can range from 0 dimensions (for the basic Γi) to any number of 
        dimensions for parallel computations. Sizes of the dimensions `...` may differ between the inputs, but most be 
        broadcast-able.

        :param sim_source: Source similarity matrix `σs` in the case base. 
        Size: `[..., M, M]`. Coordinates: `σs[a,b] = σs(a,b) ∀a,b ∈ CB`.
        :param sim_outcome: Outcome similarity matrix `σr` in the case base. 
        Size: `[..., M, M]`. Coordinates: `σr[a,b] = σr(a,b) ∀a,b ∈ CB`.

        :return: The competence of `CB` w.r.t the case `i`.
        """

        # rename in a short manner
        sim_source, new_sim_source = pop_index(sim_source, i, dim=-1)  # [..., M+1, M], [..., M+1]
        sim_outcome, new_sim_outcome = pop_index(sim_outcome, i, dim=-1)  # [..., M+1, M], [..., M+1]
        sim_source = pop_index(sim_source, i, dim=-2)[0]  # [..., M, M]
        sim_outcome = pop_index(sim_outcome, i, dim=-2)[0]  # [..., M, M]
        new_sim_source, reflexive_sim_source=pop_index(new_sim_source, i, dim=-1)  # [..., M], [...]
        new_sim_outcome, reflexive_sim_outcome=pop_index(new_sim_outcome, i, dim=-1)  # [..., M], [...]

        return MeATCubeEnergyComputations._gamma_i(
            sim_source, sim_outcome,
            new_sim_source, new_sim_outcome,
            reflexive_sim_source, reflexive_sim_outcome)
    
    @staticmethod
    def _cube_gamma_i_included(cube: torch.Tensor, i: int) -> torch.Tensor:
        """The idea is to return only the inversions at a certain index.
        
        For a base `CB` of `M` cases and a new case `i`, computes the inversions involving `i`.
        Generalizes for any dimensions `...`, that can range from 0 dimensions (for the basic Γi) to any number of 
        dimensions for parallel computations. Sizes of the dimensions `...` may differ between the inputs, but most be 
        broadcast-able.

        :param cube: Inversion cube. 
        Size: `[..., M, M, M]`.
        :param normalize: (Deprecated) If True, will normalize the competence by the cube of the CB size.

        :return: The competence of `CB` w.r.t the case `i`.
        """
        inversions = (
            (cube.select(index=i, dim=-1).sum(dim=[-1,-2]) + # gamma_abi
             cube.select(index=i, dim=-2).sum(dim=[-1,-2]) + # gamma_aic
             cube.select(index=i, dim=-3).sum(dim=[-1,-2]))  # gamma_ibc
             - (# [cannot invert itself] cube.select(i, dim=-1).select(i, dim=-1).sum(dim=-1, type=int) + # gamma_aii
                cube.select(index=i, dim=-1).select(index=i, dim=-2).sum(dim=-1) + # gamma_ibi
                cube.select(index=i, dim=-2).select(index=i, dim=-2).sum(dim=-1))  # gamma_iic
            # [cannot invert itself] - cube.select(i, dim=-1).select(i, dim=-1).select(i, dim=-1) # gamma_iii
        )

        return inversions
        
    @staticmethod
    def normalize(inversions: int, cb_size: int):
        (inversions).pow(1./3) / (cb_size.size(-1))