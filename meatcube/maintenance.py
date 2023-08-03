"""Contains the methods to remove, add, or distill cases for CB maintenance"""
import torch
from typing import Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List

try:
    from .meatcubecb import MeATCubeCB, SourceSpaceElement, OutcomeSpaceElement, NORMALIZE
    from .metrics import f1_score, accuracy
except ImportError as e:
    try:
        from meatcubecb import MeATCubeCB, SourceSpaceElement, OutcomeSpaceElement, NORMALIZE
        from metrics import f1_score, accuracy
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
    result_cb = cb.add(candidate_cases_sources[result_index], candidate_cases_outcomes[result_index])
    if k > 1:
        result_index = candidate_ranking[:k].cpu().tolist()
        for i in candidate_ranking[1:k]:
            result_cb = result_cb.add(candidate_cases_sources[i], candidate_cases_outcomes[i])

    if return_all:
        return result_cb, competences, result_index
    else:
        return result_cb

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
    competences = cb.case_competences(
        test_cases_sources=test_cases_sources,
        test_cases_outcomes=test_cases_outcomes,
        strategy=strategy,
        margin=margin,
        aggregation=None,
        normalize=normalize)
    # aggregate the results
    if aggregation == "sum":
        competences = competences.sum(dim = -1, dtype=float)
    elif aggregation == "mean":
        competences = competences.mean(dim = -1, dtype=float)
    
    # rank the candidates based on the competence
    candidate_ranking = competences.argsort()
    
    # remove 
    if k > 1:
        result_index = sorted(candidate_ranking[:k].cpu().tolist(), reverse=True)
        result_cb = cb.remove(result_index[0])
        for i in result_index[1:]:
            result_cb = result_cb.remove(i)
    else:
        result_index = candidate_ranking[0]
        result_cb = cb.remove(result_index)

    if return_all:
        return result_cb, competences, result_index
    else:
        return result_cb

MAX_COMPARATOR = lambda x, y, min_delta: x-min_delta >= y
MIN_COMPARATOR = lambda x, y, min_delta: x+min_delta <= y
MAINTENANCE_MONITORABLE_MEASURES = {
    "CB size": {
        "comparator": MIN_COMPARATOR,
        "measure": lambda cb, test_cases_sources, test_cases_outcomes: len(cb)
    },
    "F1": {
        "comparator": MAX_COMPARATOR,
        "measure": lambda cb, test_cases_sources, test_cases_outcomes: f1_score(cb, test_cases_sources, test_cases_outcomes)
    },
    "accuracy": {
        "comparator": MAX_COMPARATOR,
        "measure": lambda cb, test_cases_sources, test_cases_outcomes: accuracy(cb, test_cases_sources, test_cases_outcomes)
    },
    "hinge": {
        "comparator": MAX_COMPARATOR,
        "measure": lambda cb, test_cases_sources, test_cases_outcomes: cb.competence(test_cases_sources, test_cases_outcomes)
    }
}
def decrement_early_stopping(cb: MeATCubeCB,
                test_cases_sources: Iterable[SourceSpaceElement],
                test_cases_outcomes: Iterable[OutcomeSpaceElement],
                strategy: Literal["MCE", "hinge"]="hinge",
                margin: float=0.1,
                normalize=NORMALIZE,
                aggregation: Literal["sum", "mean"]="mean",
                step_size=1,
                return_all=True,
                monitor: Literal["F1", "accuracy", "hinge", "CB size"]="F1",
                register: Union[
                    Literal["F1", "accuracy", "hinge", "CB size"],
                    Iterable[Literal["F1", "accuracy", "hinge", "CB size"]]
                    ]="F1",
                min_delta: float=0.,
                patience: int=3,
                verbose=False,
                tqdm=False):
    """Iterative decremental process that stops using an early stopping heuristic.
    
    :param monitor: performance measure to be monitored
    :param min_delta:  minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement
    :param patience: number of steps with no improvement after which the process will be stopped

    """
    def registered_measures(cb):
        """Compute the performance measures to register"""
        if isinstance(register, Iterable):
            return {
                register_: MAINTENANCE_MONITORABLE_MEASURES[register_]["measure"](cb, test_cases_sources, test_cases_outcomes)
                for register_ in register
            }
        else:
            return {register: MAINTENANCE_MONITORABLE_MEASURES[register]["measure"](cb, test_cases_sources, test_cases_outcomes)}
    best_perf = MAINTENANCE_MONITORABLE_MEASURES[monitor]["measure"](cb, test_cases_sources, test_cases_outcomes)
    best_perf_step = 0
    records = [{
        "cb": cb,
        "step": 0,
        "cb_size": len(cb),
        "cases_removed": None, # index relative to CB from last step
        "all_cases_removed": None, # absolute index, relative to first state of the CB
        "competences_before_removal": None,
        **registered_measures(cb),
    }]
    cases = list(range(len(cb)))
    removed = []

    for step_id in range(1, len(cb)//step_size):
        cb, competences, result_index = decrement(cb,
                test_cases_sources,
                test_cases_outcomes,
                strategy=strategy,
                margin=margin,
                normalize=normalize,
                aggregation=aggregation,
                k=step_size,
                return_all=True) 

        if step_size > 1:
            removed.extend(cases[idx] for idx in result_index)
            cases = [case for idx, case in enumerate(cases) if idx not in result_index]
        else:
            removed.append(cases[result_index])
            cases.pop(result_index)
        
        perf = MAINTENANCE_MONITORABLE_MEASURES[monitor]["measure"](cb, test_cases_sources, test_cases_outcomes)
        records.append({
            "cb": cb,
            "step": step_id,
            "cb_size": len(cb),
            "cases_removed": result_index,
            "all_cases_removed": [*removed], # copy the list as it is modified at each step
            "competences_before_removal": competences,
            **registered_measures(cb),
        })

        if MAINTENANCE_MONITORABLE_MEASURES[monitor]["comparator"](perf, best_perf, min_delta):
            best_perf=perf
            best_perf_step=step_id
        elif step_id - best_perf_step >= patience:
            break

    if return_all:
        return records[best_perf_step]["cb"], records, records[best_perf_step]
    else:
        return records[best_perf_step]["cb"]