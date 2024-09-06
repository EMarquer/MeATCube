import torch
import numpy as np
import pandas as pd
from typing import Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List
from scipy.spatial.distance import squareform, pdist, cdist
from tqdm.auto import tqdm

NORMALIZE = False

# def batching_utils(batch_size=0, dim = 0, use_tqdm=False, tqdm_args={}):
#     batches_results = []
#     index_batches = [index[i:i+batch_size] for i in range(0, len(index), batch_size)]
#     for index_batch in (
#             tqdm(index_batches, **tqdm_args) if use_tqdm else
#             index_batches
#         ):
                


def pop_index(matrix: torch.Tensor, i, dim=0) -> torch.Tensor:
    dim = (dim + matrix.dim()) % matrix.dim()
    slice_base = [slice(None)]*dim
    slice_start = slice_base + [slice(i)]
    slice_excluded = slice_base + [int(i)]
    slice_end = slice_base + [slice(i+1, None)]
    return torch.cat((matrix[slice_start], matrix[slice_end]), dim=dim), matrix[slice_excluded]

def remove_index(matrix: torch.Tensor, i, dims: List[int]) -> torch.Tensor:
    for dim in dims:
        matrix = pop_index(matrix, i, dim=dim)[0]
    return matrix

def append_symmetric(symmetric_matrix: torch.Tensor, vect: torch.Tensor, cell: torch.Tensor):
    vect = vect.view(-1)
    cell = cell.view(-1)

    # from [n, n] to [n, n+1]
    symmetric_matrix = torch.cat([symmetric_matrix, vect.view(-1, 1)], dim=-1)
     # from [n] to [n+1]: add the symmetric component of the vector where the diagonal will be
    vect = torch.cat([vect, cell], dim=-1)
    # from [n, n+1] to [n+1, n+1]
    symmetric_matrix = torch.cat([symmetric_matrix, vect.view(1, -1)], dim=-2)
    
    return symmetric_matrix

# def torch_cdist():
#     pass

def pairwise_dist(
        data: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
        metric,
        **metric_kwargs):
    """A wrapper for scipy.spatial.distance.pdist, which handles a torch-based version and an object version."""
    if isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
        if (data.dtype != object):
            return pdist(data.reshape(-1,data[0].size), metric=metric)
        else:
            n = data.shape[0]
            out_size = (n * (n - 1)) // 2
            dm = np.ndarray(dtype=np.double, shape=(out_size,))
            k = 0
            for i in range(data.shape[0] - 1):
                for j in range(i + 1, data.shape[0]):
                    dm[k] = metric(data[i], data[j], **metric_kwargs)
                    k += 1
            return dm
    elif isinstance(data, torch.Tensor):
        raise NotImplementedError
    else:
        raise ValueError
def cart_dist(a: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
              b: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
              metric, **metric_kwargs):
    """A wrapper for scipy.spatial.distance.cdist, which handles a torch-based version and an object version."""
    if isinstance(a, (np.ndarray, pd.DataFrame, pd.Series)) and isinstance(b, (np.ndarray, pd.DataFrame, pd.Series)):
        if (a.dtype != object) and (b.dtype != object):
            return cdist(
                a.reshape(-1, a[0].size),
                b.reshape(-1, a[0].size),
                metric=metric)
        else:
            n = a.shape[0]
            m = b.shape[0]
            dm = np.ndarray(dtype=np.double, shape=(n, m))
            for i in range(n - 1):
                for j in range(m - 1):
                    dm[i,j] = metric(a[i], b[j], **metric_kwargs)
            return dm
    elif isinstance(a, torch.Tensor):
        raise NotImplementedError
    else:
        raise ValueError
# def pairwise_dist(
#         data1: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
#         data2: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
#         metric):
#     """A wrapper for scipy.spatial.distance.cdist"""
#     pass


class MeATCubeEnergyComputations(object):
    """(Me)asure of the complexity of a dataset for (A)nalogical (T)ransfer using Boolean (Cube)s, or slices of them."""
    @staticmethod
    def _energy(cube, normalize=NORMALIZE):
        """The energy of the case base is the """
        e = cube.sum(dim=[-3,-2,-1])
        if normalize:
            return (e).pow(1./3) / (cube.size(-1))
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
                     (oi.unsqueeze(-1) < oi.unsqueeze(-2)))
        # gamma_aic: [..., M, 1] . [..., M, M] -> [..., M, M]
        inv_aic = ((si.unsqueeze(-1) >= s) &
                     (oi.unsqueeze(-1) < o))
        # gamma_abi: [..., M, M] . [..., M, 1] -> [..., M, M]
        inv_abi = ((s >= si.unsqueeze(-1)) &
                     (o < oi.unsqueeze(-1)))
        
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
            inv_aii = torch.zeros_like(inv_iic)
            inv_iii = torch.zeros_like(inv_iic.select(-1, 0))

        return inv_ibc, inv_aic, inv_abi, inv_aii, inv_ibi, inv_iic, inv_iii

    @staticmethod
    def _gamma_i(sim_source: torch.Tensor, 
                 sim_outcome: torch.Tensor, 
                 new_sim_source: torch.Tensor,
                 new_sim_outcome: torch.Tensor,
                 reflexive_sim_source: Union[float, torch.Tensor]=1, 
                 reflexive_sim_outcome: Union[float, torch.Tensor]=1, 
                 normalize=NORMALIZE) -> torch.Tensor:
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
            exclude_impossible=True)

        # gamma_ibc, gamma_aic, gamma_abi: [..., M, M] -> [...]
        gamma_ibc = inv_ibc.sum(dim=[-2,-1]) 
        gamma_aic = inv_aic.sum(dim=[-2,-1]) 
        gamma_abi = inv_abi.sum(dim=[-2,-1]) 
        
        # gamma_ibi, gamma_iic: [..., M] -> [...]
        gamma_ibi = inv_ibi.sum(dim=-1)
        gamma_iic = inv_iic.sum(dim=-1)
        
        gamma_aii = 0 # cannot invert itself
        gamma_iii = 0 # cannot invert itself, special case of gamma_aii

        if normalize:
            return (gamma_ibc + gamma_aic + gamma_abi + gamma_aii + gamma_ibi + gamma_iic + gamma_iii).pow(1./3) / (sim_source.size(-1) + 1)
        else:
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
    def _cube_gamma_i_included(cube: torch.Tensor, i: int, normalize=NORMALIZE) -> torch.Tensor:
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

        if normalize:
            return (inversions).pow(1./3) / (cube.size(-1))
        else:
            return inversions
        