import torch


import torch
import numpy as np
import pandas as pd
from typing import Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List
from scipy.spatial.distance import squareform, pdist, cdist
from tqdm.auto import tqdm
from ..torch_utils import pop_index
from ..utils import estimate_mem_utilisation


class CtCoATEnergyComputations(object):
    @staticmethod
    def _energy(cube):
        """The energy of the case base is the """
        e = cube.sum(dim=[-3,-2,-1])
        return CtCoATEnergyComputations.normalize(e, cube.size(-1))

    @staticmethod
    def _inversion_cube(source_sim: torch.Tensor, outcome_sim: torch.Tensor, source_cube: Optional[torch.Tensor]=None,
                   outcome_cube: Optional[torch.Tensor]=None, return_all=False) -> Union[
                       torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]]:
        """Computation of the cube of energies for the Γ indicator.
        
        :param source_sim:  Tensor with coordinates `[a, b]` for `σs(a,b)`.
        :param outcome_sim: Tensor with coordinates `[a, b]` for `σr(a,b)`.
        :param return_all: If true, instead of returning only the inversion cube, returns the comparison cube for the 
        source and the outcome similarities.
        :return: Float tensor with coordinates `[a,b,c]` for `1 - (σs(a,b) - σs(a,c)*(σr(a,b) - σr(a,c))`.

        -------
        Memory usage
        -------
        3 * _cubify: source_cube, outcome_cube, and cube that takes the same size
            
        i.e., batch_dims * 2 * [M, M, M] * sim_matrix.dtype + batch_dims * [M, M, M] * sim_matrix.float
        """
        # compute only if necessary
        if source_cube is None: source_cube = CtCoATEnergyComputations._cubify(source_sim)
        if outcome_cube is None: outcome_cube = CtCoATEnergyComputations._cubify(outcome_sim)

        if source_cube.dim() == 3 & outcome_cube.dim() == 4: source_cube.unsqueeze(0)
        elif source_cube.dim() == 4 & outcome_cube.dim() == 3: outcome_cube.unsqueeze(0)
        cube = (1 - torch.mul(source_cube, outcome_cube))/2

        if return_all:
            return cube, source_cube, outcome_cube
        else:
            return cube

    @staticmethod
    def _cubify(sim_matrix: torch.Tensor) -> torch.BoolTensor:
        """Transform the similarity matrix into a cube (or an array of cubes), based on a provided comparator.

        If `sim_matrix` is of size `[M, M]`, `M` the number of cases, it should use the coordinates `[a, b]` for the 
        similarity `σ(a, b)`. Then, the output will be of size `[M, M, M]` with coordinates `[a, b, c]` for:
        - `σ(a, b) - σ(a, c)`.

        If `sim_matrix` is of size `[O, M, M]`, `M` the number of cases and `O` the number of outcomes, it should use 
        the coordinates `[o, a, b]` for the similarity :math:`σ(a, b)` given the outcome `o`. Then, the output will be
        of size `[M, M, M]` with coordinates `[o, a, b, c]` for:
        - `σ(a, b) - σ(a, c)` given the outcome `o`.

        Generalizes for shape `[..., M, M]` with `...` any number of dimensions.

        :param sim_matrix: Tensor of the similarity matrix or stack of similarity matrices.
        :return: Tensor containing the cube (or an array of cubes) of boolean values.

        -------
        Memory usage
        -------
        batch_dims * [M, M, M] * sim_matrix.dtype
        """
        if sim_matrix.dim() >= 2 and sim_matrix.size(-1) == sim_matrix.size(-1):
            # sim_matrix: [M, M]
            # cube: [M, M, M], coordinates [a, b, c], `a` the anchor
            # works by reshaping sim_matrix as follows
            # [dim anchor, dim a, . ] - [dim anchor, . , dim b]
            cube = sim_matrix.unsqueeze(-1) - sim_matrix.unsqueeze(-2)
        else:
            raise ValueError(f"Unsupported cubification of a matrix of size {sim_matrix.size()}: only works with at" 
                             "least 2 dimensions, with the last two of equal size.")
        
        return cube
    
    @staticmethod
    def _estimate_memory_energies_i(sim_source, sim_outcome, new_sim_source, new_sim_outcome,
                      reflexive_sim_source=1, reflexive_sim_outcome=1, exclude_impossible=True) -> int:
        """
        -------
        Memory usage
        -------
        batch_dims * (3 * ([M, M] + [M]) + 1) * float
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
        mem_estimate_inv_ibc = mem_estimate_inv_aic = mem_estimate_inv_abi = estimate_mem_utilisation(dtype=float, size=list(batch_sizes) + [M, M])
        mem_estimate_inv_iic = mem_estimate_inv_ibi = estimate_mem_utilisation(dtype=float, size=list(batch_sizes) + [M])
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
    def _energies_i(sim_source: torch.Tensor, sim_outcome: torch.Tensor, new_sim_source: torch.Tensor, new_sim_outcome: torch.Tensor,
                      reflexive_sim_source=1, reflexive_sim_outcome=1, exclude_impossible=False) -> (
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):

        """The idea is to compute the energies for every possible outcome, and make the choice of the right one only afterwards.
        
        For a base `CB` of `M` cases and a new case `i`, computes the energies involving `i`.
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

        """
        E(st, rt) = 
           ∑_(i,j,k ∈ CB ∪ {(st, rt)}) (1 - [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)])/2 
           - ∑_(i,j,k ∈ CB) (1 - [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)])/2
        
        with CB' = ( CB² x {(si, ri)} ) ∪ ( CB x {(si, ri)} x CB ) ∪ ( {(si, ri)} x CB²)
        E(si, ri) = 
           ∑_(a,b,c ∈ CB') (1 - [σS(sa,sb) - σS(sa,sc)]*[σR(ra,rb)-σR(ra,rc)])/2

        = ∑_(a,b ∈ CB) (1 - [σS(sa,sb) - σS(sa,si)]*[σR(ra,rb)-σR(ra,ri)])/2
        + ∑_(b,c ∈ CB) (1 - [σS(si,sb) - σS(si,sc)]*[σR(ri,rb)-σR(ri,rc)])/2
        + ∑_(a,c ∈ CB) (1 - [σS(sa,si) - σS(sa,sc)]*[σR(ra,ri)-σR(ra,rc)])/2
        + ∑_(a ∈ CB) (1 - [σS(sa,si) - σS(sa,si)]*[σR(ra,ri)-σR(ra,ri)])/2
        + ∑_(b ∈ CB) (1 - [σS(si,sb) - σS(si,si)]*[σR(ri,rb)-σR(ri,ri)])/2
        + ∑_(c ∈ CB) (1 - [σS(si,si) - σS(si,sc)]*[σR(ri,ri)-σR(ri,rc)])/2
        + (1 - [σS(si,si) - σS(si,si)]*[σR(ri,ri)-σR(ri,ri)])/2

        In particular:
            inv_abi = ∑_(a,b ∈ CB) (1 - [σS(sa,sb) - σS(sa,si)]*[σR(ra,rb)-σR(ra,ri)])/2
            inv_ibc = ∑_(b,c ∈ CB) (1 - [σS(si,sb) - σS(si,sc)]*[σR(ri,rb)-σR(ri,rc)])/2
            inv_aic = ∑_(a,c ∈ CB) (1 - [σS(sa,si) - σS(sa,sc)]*[σR(ra,ri)-σR(ra,rc)])/2

            inv_aii = ∑_(a ∈ CB) (1 - [σS(sa,si) - σS(sa,si)]*[σR(ra,ri)-σR(ra,ri)])/2
                    = ∑_(a ∈ CB) (1 - [0]*[0])/2
                    = ∑_(a ∈ CB) 1/2
            inv_ibi = ∑_(b ∈ CB) (1 - [σS(si,sb) - σS(si,si)]*[σR(ri,rb)-σR(ri,ri)])/2
                    = ∑_(b ∈ CB) (1 - [σS(si,sb) - 1]*[σR(ri,rb) - 1])/2
            inv_iic = ∑_(c ∈ CB) (1 - [σS(si,si) - σS(si,sc)]*[σR(ri,ri)-σR(ri,rc)])/2
                    = ∑_(c ∈ CB) (1 - [1 - σS(si,sc)]*[1-σR(ri,rc)])/2

            inv_iii = (1 - [σS(si,si) - σS(si,si)]*[σR(ri,ri)-σR(ri,ri)])/2
                    = (1 - [0]*[0])/2
                    = 1/2
        """


        # rename in a short manner
        s=sim_source  # [..., M, M]
        o=sim_outcome  # [..., M, M]
        si=new_sim_source  # [..., M]
        oi=new_sim_outcome  # [..., M]
        sii=reflexive_sim_source  # [...] or []
        oii=reflexive_sim_outcome  # [...] or []

        formula = lambda delta_s, delta_o: (1 - torch.mul(delta_s, delta_o))/2

        # gamma_ibc: [..., M, 1] . [..., 1, M] -> [..., M, M]
        inv_ibc = formula(
            delta_s=(si.unsqueeze(-1) - si.unsqueeze(-2)),
            delta_o=(oi.unsqueeze(-1) - oi.unsqueeze(-2)))
        # gamma_aic: [..., M, 1] . [..., M, M] -> [..., M, M]
        inv_aic = formula(
            delta_s=(si.unsqueeze(-1) - s),
            delta_o=(oi.unsqueeze(-1) - o))
        # gamma_abi: [..., M, M] . [..., M, 1] -> [..., M, M]
        inv_abi = formula(
            delta_s=(s - si.unsqueeze(-1)),
            delta_o=(o - oi.unsqueeze(-1)))
        
        if isinstance(sii, torch.Tensor) and sii.dim() == si.dim() - 1:
            sii = sii.unsqueeze(-1)
        if isinstance(oii, torch.Tensor) and oii.dim() == oi.dim() - 1:
            oii = oii.unsqueeze(-1)
        # gamma_ibi: [..., M] . [] -> [..., M]
        inv_ibi = formula(
            delta_s=(si - sii),
            delta_o=(oi - oii))
        # gamma_iic: [] . [..., M] -> [..., M]
        inv_iic = formula(
            delta_s=(sii - si), 
            delta_o=(oii - oi))
        
        if not exclude_impossible:
            inv_aii = torch.ones_like(inv_iic)/2
            inv_iii = torch.ones_like(inv_iic.select(-1, 0))/2
        else:
            inv_aii = 1/2
            inv_iii = 1/2

        return inv_ibc, inv_aic, inv_abi, inv_aii, inv_ibi, inv_iic, inv_iii

    @staticmethod
    def _estimate_memory_gamma_i(sim_source, sim_outcome, new_sim_source, new_sim_outcome,
                      reflexive_sim_source=1, reflexive_sim_outcome=1) -> int:
        """

        -------
        Memory usage
        -------
        _energies_i + (6 * batch_dims) * float

        i.e. batch_dims * (3 * [M, M] + 2 * [M] + 7) * float
        """
        # get max broadcast sizes
        batch_sizes = list(sim_source.size()[:-2])
        other_inputs = [sim_outcome.size()[:-2], new_sim_source.size()[:-1], new_sim_outcome.size()[:-1]]
        if isinstance(reflexive_sim_source, torch.Tensor): other_inputs += [reflexive_sim_source.size()]
        if isinstance(reflexive_sim_outcome, torch.Tensor): other_inputs += [reflexive_sim_outcome.size()]
        for other_batch_sizes in other_inputs:
            for i in range(len(batch_sizes)):
                batch_sizes[i] = max(batch_sizes[i], other_batch_sizes[i])
        
        memory_inversions_i = CtCoATEnergyComputations._estimate_memory_energies_i(
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
        """The idea is to compute the energies for every possible outcome, and make the choice of the right one only afterwards.
        
        For a base `CB` of `M` cases and a new case `i`, computes the energies involving `i`.
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
        inv_ibc, inv_aic, inv_abi, inv_aii, inv_ibi, inv_iic, inv_iii = CtCoATEnergyComputations._energies_i(
            sim_source, sim_outcome, # [..., M, M]
            new_sim_source, new_sim_outcome, # [..., M]
            reflexive_sim_source=reflexive_sim_source, reflexive_sim_outcome=reflexive_sim_outcome, # [...] or []
            exclude_impossible=True,
            )

        # gamma_ibc, gamma_aic, gamma_abi: [..., M, M] -> [...]
        gamma_ibc = inv_ibc.sum(dim=[-2,-1]) 
        gamma_aic = inv_aic.sum(dim=[-2,-1]) 
        gamma_abi = inv_abi.sum(dim=[-2,-1]) 
        
        # gamma_ibi, gamma_iic: [..., M] -> [...]
        gamma_ibi = inv_ibi.sum(dim=-1)
        gamma_iic = inv_iic.sum(dim=-1)
        gamma_aii = inv_iic.size(-1) * 1/2 #inv_aii.sum(dim=-1) # cannot invert itself
        gamma_iii = 1/2 # cannot invert itself, special case of gamma_aii

        return gamma_ibc + gamma_aic + gamma_abi + gamma_aii + gamma_ibi + gamma_iic + gamma_iii
    
    @staticmethod
    def _gamma_i_included(sim_source, sim_outcome, i) -> torch.Tensor:
        """The idea is to compute the energies for every possible outcome, and make the choice of the right one only afterwards.
        
        For a base `CB` of `M` cases and a new case `i`, computes the energies involving `i`.
        Generalizes for any dimensions `...`, that can range from 0 dimensions (for the basic Γi) to any number of 
        dimensions for parallel computations. Sizes of the dimensions `...` may differ between the inputs, but most be 
        broadcast-able.

        :param sim_source: Source similarity matrix `σs` in the case base. 
        Size: `[..., M, M]`. Coordinates: `σs[a,b] = σs(a,b) ∀a,b ∈ CB`.
        :param sim_outcome: Outcome similarity matrix `σr` in the case base. 
        Size: `[..., M, M]`. Coordinates: `σr[a,b] = σr(a,b) ∀a,b ∈ CB`.

        :return: The competence of `CB` w.r.t the case `i`.

        -------
        Memory usage
        -------
        _energies_i + (6 * batch_dims) * float

        i.e. batch_dims * (3 * ([M-1, M-1] + [M-1]) + 7) * float
        """

        # rename in a short manner
        sim_source, new_sim_source = pop_index(sim_source, i, dim=-1)  # [..., M-1, M], [..., M]
        sim_outcome, new_sim_outcome = pop_index(sim_outcome, i, dim=-1)  # [..., M-1, M], [..., M]
        sim_source = pop_index(sim_source, i, dim=-2)[0]  # [..., M-1, M-1]
        sim_outcome = pop_index(sim_outcome, i, dim=-2)[0]  # [..., M-1, M-1]
        new_sim_source, reflexive_sim_source=pop_index(new_sim_source, i, dim=-1)  # [..., M-1], [...]
        new_sim_outcome, reflexive_sim_outcome=pop_index(new_sim_outcome, i, dim=-1)  # [..., M-1], [...]

        return CtCoATEnergyComputations._gamma_i(
            sim_source, sim_outcome,
            new_sim_source, new_sim_outcome,
            reflexive_sim_source, reflexive_sim_outcome)
    
    @staticmethod
    def _cube_gamma_i_included(cube: torch.Tensor, i: int) -> torch.Tensor:
        """The idea is to return only the energies at a certain index.
        
        For a base `CB` of `M` cases and a new case `i`, computes the energies involving `i`.
        Generalizes for any dimensions `...`, that can range from 0 dimensions (for the basic Γi) to any number of 
        dimensions for parallel computations. Sizes of the dimensions `...` may differ between the inputs, but most be 
        broadcast-able.

        :param cube: Inversion cube. 
        Size: `[..., M, M, M]`.
        :param normalize: (Deprecated) If True, will normalize the competence by the cube of the CB size.

        :return: The competence of `CB` w.r.t the case `i`.

        -------
        Memory usage
        -------
        i.e. batch_dims * (3 * [M, M] + 2 * [M]) * float
        """
        energies = (
            (cube.select(index=i, dim=-1).sum(dim=[-1,-2]) + # gamma_abi
             cube.select(index=i, dim=-2).sum(dim=[-1,-2]) + # gamma_aic
             cube.select(index=i, dim=-3).sum(dim=[-1,-2]))  # gamma_ibc
             - (# [cannot invert itself] cube.select(i, dim=-1).select(i, dim=-1).sum(dim=-1, type=int) + # gamma_aii
                cube.select(index=i, dim=-1).select(index=i, dim=-2).sum(dim=-1) + # gamma_ibi
                cube.select(index=i, dim=-2).select(index=i, dim=-2).sum(dim=-1))  # gamma_iic
            # [cannot invert itself] - cube.select(i, dim=-1).select(i, dim=-1).select(i, dim=-1) # gamma_iii
        )

        return energies
        
    @staticmethod
    def normalize(e: int, cb_size: int):

        # # (e + (n^2)/2)/(n^3) = e/(n^3) + ((n^2)/2)/(n^3) = e/(n^3) + (1/2n)
        # #e = (e + (cube.size(-1)^2)/2)/(cube.size(-1)^3) # (e + (n^2)/2)/(n^3)
        # e = (e/(cb_size**3)) + (1/(2*cb_size)) # e/(n^3) + (1/2n)

        return e