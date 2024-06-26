"""Contains the methods to remove, add, or distill cases for CB maintenance"""
import torch
import numpy as np
from typing import Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List

try:
    from .meatcubecb import MeATCubeCB, SourceSpaceElement, OutcomeSpaceElement, NORMALIZE
    from .torch_backend import MeATCubeEnergyComputations
    from .metrics import f1_score, accuracy
except ImportError as e:
    try:
        from meatcubecb import MeATCubeCB, SourceSpaceElement, OutcomeSpaceElement, NORMALIZE
        from torch_backend import MeATCubeEnergyComputations, append_symmetric
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
    """Compute the competence of the CB w.r.t each candidate, then add the `k` for which the CB is the most incompetent.
    
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
    

def increment_(cb: MeATCubeCB,
                candidate_cases_sources: Iterable[SourceSpaceElement],
                candidate_cases_outcomes: Iterable[OutcomeSpaceElement],
                test_cases_sources: Iterable[SourceSpaceElement],
                test_cases_outcomes: Iterable[OutcomeSpaceElement],
                strategy: Literal["MCE", "hinge"]="hinge",
                margin: float=0.1,
                normalize=NORMALIZE,
                return_all=True) -> Union[MeATCubeCB, Tuple[MeATCubeCB, torch.Tensor, Union[int, List[int]]]]:
    """Compute the competence of the CB w.r.t the test set after adding each candidate, then add the case improving the competence the most.
    If no case improve the competence, nothing is added.
    (Naive implementation)

    :return: If `return_all`, returns the updated CB, the competence w.r.t each candidate case, and the index 
        (if `k`==1) or list of indices (if `k`>=1) of the candidate(s) that was added.
    """

    best_competence = cb.competence(
            test_cases_sources=test_cases_sources,
            test_cases_outcomes=test_cases_outcomes,
            strategy=strategy,
            margin=margin,
            aggregation="mean",
            normalize=normalize)
    best_candidate = None
    for case_id, (candidate_case_source, candidate_case_outcome) in enumerate(zip(candidate_cases_sources, candidate_cases_outcomes)):
        cb_ = cb.add(candidate_case_source, candidate_case_outcome)
        competence = cb_.competence(
            test_cases_sources=test_cases_sources,
            test_cases_outcomes=test_cases_outcomes,
            strategy=strategy,
            margin=margin,
            aggregation="mean",
            normalize=normalize)
        if competence > best_competence:
            best_candidate = {
                "case_id": case_id,
                "case": (candidate_case_source, candidate_case_outcome),
                "cb": cb_,
                "competence": competence
            }
            best_competence = competence
    
    if return_all:
        return best_candidate
    else:
        return None if best_candidate is None else best_candidate["cb"]

def decrement(cb: MeATCubeCB,
                test_cases_sources: Iterable[SourceSpaceElement],
                test_cases_outcomes: Iterable[OutcomeSpaceElement],
                strategy: Literal["MCE", "hinge"]="hinge",
                margin: float=0.1,
                normalize=NORMALIZE,
                aggregation: Literal["sum", "mean"]="mean",
                k=1,
                return_all=True,
                batch_size=0) -> Union[MeATCubeCB, Tuple[MeATCubeCB, torch.Tensor, Union[int, List[int]]]]:
    """Compute the competence of the CB w.r.t the test set, then remove the `k` cases participating the least to 
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
        normalize=normalize,
        batch_size=batch_size)
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
                batch_size=0,
                verbose=False,
                tqdm=False):
    """Iterative decremental process that stops using an early stopping heuristic.
    
    :param monitor: performance measure to be monitored
    :param min_delta:  minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement
    :param patience: number of steps with no improvement after which the process will be stopped

    """
    def registered_measures(cb):
        """Compute the performance measures to register"""
        if not isinstance(register, str) and isinstance(register, Iterable):
            return {
                register_: MAINTENANCE_MONITORABLE_MEASURES[register_]["measure"](cb, test_cases_sources, test_cases_outcomes)
                for register_ in register
            }
        else:
            return {register: MAINTENANCE_MONITORABLE_MEASURES[register]["measure"](cb, test_cases_sources, test_cases_outcomes)}
    best_perf = MAINTENANCE_MONITORABLE_MEASURES[monitor]["measure"](cb, test_cases_sources, test_cases_outcomes)
    best_perf_step = 0
    best_record = {
        "cb": cb,
        "step": 0,
        "cb_size": len(cb),
        "cases_removed": None, # index relative to CB from last step
        "all_cases_removed": None, # absolute index, relative to first state of the CB
        "competences_before_removal": None,
        **registered_measures(cb),
    }
    if return_all:
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
                return_all=True,
                batch_size=batch_size) 

        if step_size > 1:
            removed.extend(cases[idx] for idx in result_index)
            cases = [case for idx, case in enumerate(cases) if idx not in result_index]
        else:
            removed.append(cases[result_index])
            cases.pop(result_index)
        
        perf = MAINTENANCE_MONITORABLE_MEASURES[monitor]["measure"](cb, test_cases_sources, test_cases_outcomes)
        record = {
                "cb": cb,
                "step": step_id,
                "cb_size": len(cb),
                "cases_removed": result_index,
                "all_cases_removed": [*removed], # copy the list as it is modified at each step
                "competences_before_removal": competences,
                **registered_measures(cb),
            }
        if return_all:
            records.append(record)

        if MAINTENANCE_MONITORABLE_MEASURES[monitor]["comparator"](perf, best_perf, min_delta):
            best_perf=perf
            best_perf_step=step_id
            best_record=record
        elif step_id - best_perf_step >= patience:
            break

    if return_all:
        return records[best_perf_step]["cb"], records, records[best_perf_step]
    else:
        return best_record["cb"]
    

def diversity(cb: MeATCubeCB, ranks, k=1):
    """Diversity of the k lowest ranked cases"""
    subset = cb.source_sim_matrix[ranks[:k]][:,ranks[:k]]
    diversity_value = 2 * (1-subset).triu().sum() / (k * (k-1))


def diversity_decrement(cb: MeATCubeCB,
                test_cases_sources: Iterable[SourceSpaceElement],
                test_cases_outcomes: Iterable[OutcomeSpaceElement],
                strategy: Literal["MCE", "hinge"]="hinge",
                margin: float=0.1,
                normalize=NORMALIZE,
                aggregation: Literal["sum", "mean"]="mean",
                k=1,
                return_all=True) -> Union[MeATCubeCB, Tuple[MeATCubeCB, torch.Tensor, Union[int, List[int]]]]:
    """Compute the competence of the CB w.r.t the test set, then remove the `k` cases participating the least to 
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













    
def fancy_distillation_process(cb: MeATCubeCB,
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
                tqdm=False,
                max_steps=500):
    # repeat until no modification happens:
    # 1. remove cases until no improvement is observed
    # 2. add cases until no improvement is observed

    steps = 0
    new_cb = cb
    candidate_pool = []
    cases = list(range(len(cb)))
    removed = True
    added = False
    while (added or removed) and steps < max_steps:
        steps += 1
        
        # 1. remove cases until no improvement is observed
        new_cb, records, last_record = decrement_early_stopping(new_cb,
                test_cases_sources,
                test_cases_outcomes,
                strategy=strategy,
                margin=margin,
                normalize=normalize,
                aggregation=aggregation,
                step_size=step_size,
                return_all=True,
                monitor=monitor,
                register=register,
                min_delta=min_delta,
                patience=patience,
                verbose=verbose,
                tqdm=tqdm)
        removed = last_record["step"] > 0
        
        # re-aligned cases removed this iteration with their initial ID
        removed_cases = last_record["all_cases_removed"]
        removed_cases = [cases[idx] for idx in removed_cases]
        # remove from the current case indices the indices of the cases removed this iteration
        cases = [case for idx, case in enumerate(cases) if idx not in removed_cases]
        candidate_pool = sorted(set(range(len(cb))).difference(removed_cases))
        print(f"Removed {len(removed_cases)} cases at step {steps}")

        # 2. add cases until no improvement is observed
        added_cases = []
        if len(candidate_pool) > 0:
            added = True
            while added and len(candidate_pool) > 0:
                candidate_cases_sources = np.take(cb.CB_source, candidate_pool, axis=0)
                candidate_cases_outcomes = np.take(cb.CB_outcome, candidate_pool, axis=0)
                best_candidate = increment_(new_cb,
                        candidate_cases_sources,
                        candidate_cases_outcomes,
                        test_cases_sources,
                        test_cases_outcomes,
                        strategy=strategy,
                        margin=margin,
                        normalize=normalize,
                        return_all=True)
                if best_candidate is None:
                    added = False
                else:
                    new_cb = best_candidate["cb"]
                    # memorize that a new case has been added and remove it from the candidates
                    cases.append(candidate_pool.pop(best_candidate["case_id"]))
                    added_cases.append(cases[-1])
            
            print(f"Added {len(added_cases)} cases at step {steps}")
        else:
            added = False
            print(f"Could not add cases at step {steps}")
    return cb