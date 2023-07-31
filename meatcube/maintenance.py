"""Contains the methods to remove, add, or distill cases for CB maintenance"""
import torch
from typing import Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List

try:
    from .meatcubecb import MeATCubeCB, SourceSpaceElement, OutcomeSpaceElement, NORMALIZE
except ImportError as e:
    try:
        from meatcubecb import MeATCubeCB, SourceSpaceElement, OutcomeSpaceElement, NORMALIZE
    except ImportError:
        raise e

def increment(cb: MeATCubeCB,
                candidate_cases_sources: Iterable[SourceSpaceElement],
                candidate_cases_outcomes: Iterable[OutcomeSpaceElement],
                strategy: Literal["MCE", "hinge"]="hinge",
                margin: float=0.1,
                normalize=NORMALIZE,
                k=1,
                return_all=True) -> Union[MeATCubeCB, Tuple[MeATCubeCB, torch.Tensor, Union[int, List[int]]]]:
    """Compute the competence of the CB w.r.t each candidate, then add the `k` most incompetent.
    
    :return: If `return_all`, returns the updated CB, the competence w.r.t each candidate case, and the index 
        (if `k`==1) or list of indices (if `k`>=1) of the candidate(s) that was added.
    """

    competences = cb.competence(
        test_cases_sources=candidate_cases_sources,
        test_cases_outcomes=candidate_cases_outcomes,
        index=None,
        strategy=strategy,
        margin=margin,
        aggregation=None,
        normalize=normalize)
    
    # rank the candidates based on the competence
    candidate_ranking = competences.argsort()
    
    result_index = candidate_ranking[0]
    result_cube = cb.add(candidate_cases_sources[result_index], candidate_cases_outcomes[result_index])
    if k > 1:
        result_index = candidate_ranking[:k].cpu().tolist()
        for i in candidate_ranking[1:k]:
            result_cube = result_cube.add(candidate_cases_sources[i], candidate_cases_outcomes[i])

    if return_all:
        return result_cube, competences, result_index
    else:
        return result_cube

def decrement(cb: MeATCubeCB,
                test_cases_sources: Iterable[SourceSpaceElement],
                test_cases_outcomes: Iterable[OutcomeSpaceElement],
                strategy: Literal["MCE", "hinge"]="hinge",
                margin: float=0.1,
                normalize=NORMALIZE,
                aggregation: Literal["sum", "mean"]="mean",
                k=1,
                return_all=True) -> Union[MeATCubeCB, Tuple[MeATCubeCB, torch.Tensor, Union[int, List[int]]]]:
    """Compute the competence of the CB w.r.t each candidate, then remove the `k` cases participating the least to 
    the competence.
    
    :return: If `return_all`, returns the updated CB, the competence w.r.t each candidate case, and the index 
        (if `k`==1) or list of indices (if `k`>=1) of the candidate(s) that was added.
    """
    competences = cb.competence_contrib(
        test_cases_sources=test_cases_sources,
        test_cases_outcomes=test_cases_outcomes,
        strategy=strategy,
        margin=margin,
        aggregation=None,
        normalize=normalize)
    # aggregate the results
    if aggregation == "sum":
        competences = competences.sum(dim = -1)
    elif aggregation == "mean":
        competences = competences.mean(dim = -1)
    
    # rank the candidates based on the competence
    candidate_ranking = competences.argsort()
    
    # remove 
    if k > 1:
        result_index = sorted(candidate_ranking[:k].cpu().tolist(), reverse=True)
        result_cube = cb.remove(result_index[0])
        for i in result_index[1:]:
            result_cube = result_cube.remove(i)
    else:
        result_index = candidate_ranking[0]
        result_cube = cb.remove(result_index)

    if return_all:
        return result_cube, competences, result_index
    else:
        return result_cube