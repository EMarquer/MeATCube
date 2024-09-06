

from __future__ import annotations # for self-referring type hints
from typing import Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List
from collections.abc import Sequence
from abc import abstractmethod, ABC, ABCMeta
import pickle

import numpy as np
import torch
Matrix = Union[torch.Tensor, np.ndarray]

SourceSpaceElement = TypeVar('SourceSpaceElement')
OutcomeSpaceElement = TypeVar('OutcomeSpaceElement')

class ACaseBaseEnergyPredictor(Sequence, Generic[SourceSpaceElement, OutcomeSpaceElement]):
    """An abstract class for energy-based CB predictors, implementing default behavior for all competence/expertise 
    measures and for prediction.

    The minimum to define is:
    - CB manipulation (potential_outcomes, add, remove, len, getitem)
    - the energy function (energy_cb)
    
    Attributes
    ----------
    _X : Iterable[SourceSpaceElement]
        the situations of the cases in the CB
    _y : Iterable[OutcomeSpaceElement]
        the outcomes of the cases in the CB
    sim_X : (SourceSpaceElement,SourceSpaceElement) -> float
        the situation space similarity 
    sim_y : (OutcomeSpaceElement,OutcomeSpaceElement) -> float
        the outcome space similarity
    """
    
    """qsdqdq"""
    _X: Iterable[SourceSpaceElement] 
    _y: Iterable[OutcomeSpaceElement]
    sim_X: Callable[[SourceSpaceElement, SourceSpaceElement], float]
    sim_y: Callable[[OutcomeSpaceElement, OutcomeSpaceElement], float]
    
    def __init__(self,
                 sim_X: Callable[[SourceSpaceElement, SourceSpaceElement], float],
                 sim_y: Callable[[OutcomeSpaceElement, OutcomeSpaceElement], float]):
        """
        
        Parameters
        ----------
        sim_X : the similarity measure for the source space
        sim_y : the similarity measure for the outcome space
        """
        try: pickle.dumps(sim_X)
        except AttributeError: raise ValueError("sim_X not pickleable, but it should be") 
        try: pickle.dumps(sim_y)
        except AttributeError: raise ValueError("sim_y not pickleable, but it should be") 
        self.sim_X = sim_X
        self.sim_y = sim_y

    potential_outcomes_: List[OutcomeSpaceElement]

    @abstractmethod
    def add(self, case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement) -> ACaseBaseEnergyPredictor:
        """Returns a copy of this ACaseBase object where case `(case_source, case_outcome)` has been added as the last element.
        
        If initialized, will copy and update the variables linked to the case.
        """
        raise NotImplementedError()

    @abstractmethod
    def remove(self, index: int) -> ACaseBaseEnergyPredictor:
        """Returns a copy of this ACaseBase object where case at `index` has been removed.
        
        If initialized, will copy and update the variables linked to the case."""
        raise NotImplementedError()
          
    @abstractmethod  
    def __len__(self) -> int:
        """
        Returns
        -------
        length : the number of cases in this CB
        """
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index) -> Tuple[SourceSpaceElement, OutcomeSpaceElement]:
        raise NotImplementedError()


    @abstractmethod
    def energy_cb(self, **kwargs) -> float:
        """Computes the energy of the case base."""
        raise NotImplementedError()
    def energy_case_from_cb(self, index: int, **kwargs) -> float:
        """Computes the contribution of a case to the energy."""
        return self.energy_cb(**kwargs) - self.remove(index).energy_cb(**kwargs)
    def energy_case_new(self, case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement, **kwargs) -> float:
        """Computes the contribution of a case to the energy."""
        #return self.add(case_source, case_outcome).energy_cb(**kwargs) - self.energy_cb(**kwargs)
        return self.add(case_source, case_outcome).energy_case_from_cb(-1, **kwargs)
    def energy_cases_new(self, case_source: Iterable[SourceSpaceElement], case_outcome: Iterable[OutcomeSpaceElement], **kwargs) -> Matrix:
        """Computes the contribution of one or multiple cases to the energy.
        
        This function tries all possible combinations in `case_source x case_outcome`.
        
        Returns:
            A `|case_source|` by `|case_outcome|` matrix containing the energy of all possible combinations at coordinates [source, outcome].
        """
        return np.array([[self.energy_case_new(source, outcome, **kwargs) for outcome in case_outcome] for source in case_source])
    
    
    def predict_one(self, case_source: SourceSpaceElement, candidate_outcomes=None, return_outcome_id=False, return_outcome_energy=False, **kwargs) -> Union[
            OutcomeSpaceElement, Tuple[OutcomeSpaceElement, int], Tuple[OutcomeSpaceElement, float], Tuple[OutcomeSpaceElement, int, float]]:
        """Compute the energy of a new situation combined with each of `candidate_outcomes`.
        
        Parameters
        ----------
        candidate_outcomes:
            If provided, use as a list of possible outcomes, otherwise use `self.potential_outcomes`.
        return_outcome_id:
            If True, returns the outcome index corresponding to the outcome.
        return_outcome_energy:
            If True, returns the energy corresponding to the outcome.

        Returns
        -------
        outcome(, index)(, energy):
            Depends on the values of `return_outcome_id` and `return_outcome_energy`
            - if `return_outcome_id==False` and `return_outcome_energy==False`, returns the outcome only
            - if `return_outcome_id==True` and `return_outcome_energy==False`, returns the outcome and the corresponding index
            - if `return_outcome_id==False` and `return_outcome_energy==True`, returns the outcome and the corresponding energy
            - if `return_outcome_id==True` and `return_outcome_energy==True`, returns the outcome, the corresponding index, and the corresponding energy
        """
        if candidate_outcomes is None: # use self.potential_outcomes by default
            candidate_outcomes = self.potential_outcomes_

        energies = self.energy_cases_new([case_source], candidate_outcomes, **kwargs)[0]
        min_index = energies.argmin()
        if isinstance(min_index, torch.Tensor): min_index = min_index.item() # failsafe if energy_cases_new returns a tensor

        if return_outcome_id and return_outcome_energy:
            return candidate_outcomes[min_index], min_index, energies[min_index]
        elif return_outcome_id:
            return candidate_outcomes[min_index], min_index
        elif return_outcome_energy:
            return candidate_outcomes[min_index], energies[min_index]
        else:
            return candidate_outcomes[min_index]
    def predict_multiple(self, cases_sources: Iterable[SourceSpaceElement], candidate_outcomes=None, return_outcome_id=False, return_outcome_energy=False, **kwargs) -> Union[
            Iterable[OutcomeSpaceElement], Tuple[Iterable[OutcomeSpaceElement], Iterable[int]], Tuple[Iterable[OutcomeSpaceElement], Iterable[float]], Tuple[Iterable[OutcomeSpaceElement], Iterable[int], Iterable[float]]]:
        """Compute the energy of new situations combined with each of `candidate_outcomes`.
        
        Parameters
        ----------
        candidate_outcomes:
            If provided, use as a list of possible outcomes, otherwise use `self.potential_outcomes`.
        return_outcome_id:
            If True, returns the outcome index corresponding to each outcome.
        return_outcome_energy:
            If True, returns the energies corresponding to each outcome.

        Returns
        -------
        outcome(, index)(, energy):
            Depends on the values of `return_outcome_id` and `return_outcome_energy`
            - if `return_outcome_id==False` and `return_outcome_energy==False`, returns the outcome only
            - if `return_outcome_id==True` and `return_outcome_energy==False`, returns the outcome and the corresponding index
            - if `return_outcome_id==False` and `return_outcome_energy==True`, returns the outcome and the corresponding energies
            - if `return_outcome_id==True` and `return_outcome_energy==True`, returns the outcome, the corresponding index, and the corresponding energies
        """
        predictions = [self.predict_one(case_source, candidate_outcomes=candidate_outcomes, return_outcome_id=return_outcome_id, return_outcome_energy=return_outcome_energy, **kwargs) for case_source in cases_sources]
        if return_outcome_id or return_outcome_energy:
            return (list(sequence) for sequence in zip(*predictions)) # transform a list of tuples into a tuple of lists
        else:
            return predictions


    def expertise_cb(self, 
            test_case_source: SourceSpaceElement,
            test_case_outcome: OutcomeSpaceElement,
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            **kwargs) -> float:
        """Compute the competence `ℓ(CB, cₜ)` of the case base `CB` w.r.t a test case `cₜ`.
        
        :param strategy:
            If `strategy` = "MCE", use the minimum classification error loss:
            `ℓ(CB, cₜ) = - ℓmce(CB, cₜ) = - ( E(CB ∪ {(sₜ, rₜ)})) - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')})) )`.

            If `strategy` = "hinge", use the hinge competence (hinge, zero_division=0 competence = - hinge loss):
            `ℓ(CB, cₜ) = - max(0, λ + ℓmce(CB, cₜ))`.
        :param margin: 
            If `strategy` = "hinge", `margin` corresponds to the `λ` parameter.
        """
        if strategy == "MCE":
            # find `rₜ' ≠ rₜ`
            try:
                outcome_index = self.potential_outcomes_.index(test_case_outcome)
                other_outcomes = self.potential_outcomes_[:outcome_index] + self.potential_outcomes_[outcome_index+1:]
            except ValueError:
                other_outcomes = self.potential_outcomes_

            return - (
                self.energy_case_new(test_case_source, test_case_outcome, **kwargs) # E(CB ∪ {(sₜ, rₜ)}
                - self.predict_one(test_case_source, candidate_outcomes=other_outcomes, return_outcome_energy=True, **kwargs)[1] # min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')})
            )
        elif strategy == "hinge":
            return - max(0, margin - self.expertise_cb(test_case_source, test_case_outcome, strategy="MCE", **kwargs))
        else: 
            raise ValueError(f"Unexpected competence/expertise strategy: {strategy}")
    def expertise_case_from_cb(self,
            index: int, 
            test_case_source: SourceSpaceElement,
            test_case_outcome: OutcomeSpaceElement,
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            **kwargs) -> float:
        """Computes the contribution `C(c, CB, cₜ)` of a case `c ∈ CB` to the competence w.r.t a test case `cₜ`, or influence w.r.t a test case `cₜ`.
        
         `C(c, CB, cₜ) = ℓ(CB, cₜ) - ℓ(CB\{c}, cₜ)` 

        :param strategy:
            If `strategy` = "MCE", use the minimum classification error loss:
            `ℓ(CB, cₜ) = - ℓmce(CB, cₜ) = - ( E(CB ∪ {(sₜ, rₜ)})) - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')})) )`.

            If `strategy` = "hinge", use the hinge competence (hinge, zero_division=0 competence = - hinge loss):
            `ℓ(CB, cₜ) = - max(0, λ + ℓmce(CB, cₜ))`.
        :param margin: 
            If `strategy` = "hinge", `margin` corresponds to the `λ` parameter.
        """
        return (
            self.expertise_cb(test_case_source, test_case_outcome, strategy=strategy, margin=margin, **kwargs)
            - self.remove(index).expertise_cb(test_case_source, test_case_outcome, strategy=strategy, margin=margin, **kwargs)
        )
    def expertise_case_new(self,
            case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement,
            test_case_source: SourceSpaceElement,
            test_case_outcome: OutcomeSpaceElement,
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            **kwargs) -> float:
        """Computes the contribution `C(c, CB ∪ {c}, cₜ)` of a case `c` to the competence w.r.t a test case `cₜ`, or influence w.r.t a test case `cₜ`.
        
         `C(c, CB ∪ {c}, cₜ) = ℓ(CB ∪ {c}, cₜ) - ℓ(CB, cₜ)` 

        :param strategy:
            If `strategy` = "MCE", use the minimum classification error loss:
            `ℓ(CB, cₜ) = - ℓmce(CB, cₜ) = - ( E(CB ∪ {(sₜ, rₜ)})) - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')})) )`.

            If `strategy` = "hinge", use the hinge competence (hinge, zero_division=0 competence = - hinge loss):
            `ℓ(CB, cₜ) = - max(0, λ + ℓmce(CB, cₜ))`.
        :param margin: 
            If `strategy` = "hinge", `margin` corresponds to the `λ` parameter.
        """
        # return (
        #     self.add(case_source, case_outcome).expertise_cb(test_case_source, test_case_outcome, strategy=strategy, margin=margin, **kwargs)
        #     - self.expertise_cb(test_case_source, test_case_outcome, strategy=strategy, margin=margin, **kwargs)
        # )
        return self.add(case_source, case_outcome).expertise_case_from_cb(-1, test_case_source, test_case_outcome, strategy=strategy, margin=margin, **kwargs)
    def competence_cb(self,
            test_cases_sources: Iterable[SourceSpaceElement],
            test_cases_outcomes: Iterable[OutcomeSpaceElement],
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            aggregation: Literal[None, "none", "sum", "mean"]="mean",
            **kwargs) -> Union[float, List[float]]:
        """Compute the competence `C(CB, T)` of the case base `CB` w.r.t a test set `T`.

        :param aggregation:
            If `aggregation` = "mean", use the mean of expertises over `T`:
            `C(CB, T) = 1/|T| (∑_{cₜ ∈ T} ℓ(CB, cₜ))`

            If `aggregation` = "sum", use the sum of expertises over `T`:
            `C(CB, T) = ∑_{cₜ ∈ T} ℓ(CB, cₜ)`

            If `aggregation` = "none" or `aggregation` = `None`, return the individual expertises of `CB` over `T`:
            `C(CB, T) = [ℓ(CB, cₜ) for cₜ ∈ T]`
        :param strategy:
            If `strategy` = "MCE", use the minimum classification error loss:
            `ℓ(CB, cₜ) = - ℓmce(CB, cₜ) = - ( E(CB ∪ {(sₜ, rₜ)})) - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')})) )`.

            If `strategy` = "hinge", use the hinge competence (hinge, zero_division=0 competence = - hinge loss):
            `ℓ(CB, cₜ) = - max(0, λ + ℓmce(CB, cₜ))`.
        :param margin: 
            If `strategy` = "hinge", `margin` corresponds to the `λ` parameter.
        """
        expertises = [self.expertise_cb(*test_case, strategy=strategy, margin=margin, **kwargs) for test_case in zip(test_cases_sources, test_cases_outcomes)]
        if aggregation == "mean":
            return sum(expertises) / len(expertises)
        elif aggregation == "sum":
            return sum(expertises)
        else:
            return expertises
    def competence_case_from_cb(self,
            index: int,
            test_cases_sources: Iterable[SourceSpaceElement],
            test_cases_outcomes: Iterable[OutcomeSpaceElement],
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            aggregation: Literal[None, "none", "sum", "mean"]="mean",
            **kwargs) -> Union[float, List[float]]:
        """Compute the competence `C(c, CB, T)` of the case `c ∈ CB` w.r.t a test set `T`.

        :param aggregation:
            If `aggregation` = "mean", use the mean of expertises over `T`:
            `C(CB, T) = 1/|T| (∑_{cₜ ∈ T} ℓ(CB, cₜ))`

            If `aggregation` = "sum", use the sum of expertises over `T`:
            `C(CB, T) = ∑_{cₜ ∈ T} ℓ(CB, cₜ)`

            If `aggregation` = "none" or `aggregation` = `None`, return the individual expertises of `CB` over `T`:
            `C(CB, T) = [ℓ(CB, cₜ) for cₜ ∈ T]`
        :param strategy:
            If `strategy` = "MCE", use the minimum classification error loss:
            `ℓ(CB, cₜ) = - ℓmce(CB, cₜ) = - ( E(CB ∪ {(sₜ, rₜ)})) - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')})) )`.

            If `strategy` = "hinge", use the hinge competence (hinge, zero_division=0 competence = - hinge loss):
            `ℓ(CB, cₜ) = - max(0, λ + ℓmce(CB, cₜ))`.
        :param margin: 
            If `strategy` = "hinge", `margin` corresponds to the `λ` parameter.
        """
        return (
            self.competence_cb(test_cases_sources, test_cases_outcomes, strategy=strategy, margin=margin, aggregation=aggregation, **kwargs)
            - self.remove(index).competence_cb(test_cases_sources, test_cases_outcomes, strategy=strategy, margin=margin, aggregation=aggregation, **kwargs)
        )
    def competence_case_new(self,
            case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement,
            test_cases_sources: Iterable[SourceSpaceElement],
            test_cases_outcomes: Iterable[OutcomeSpaceElement],
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            aggregation: Literal[None, "none", "sum", "mean"]="mean",
            **kwargs) -> Union[float, List[float]]:
        """Compute the competence `C(c, CB ∪ {c}, T)` of the case `c` w.r.t a test set `T`.

        :param aggregation:
            If `aggregation` = "mean", use the mean of expertises over `T`:
            `C(CB, T) = 1/|T| (∑_{cₜ ∈ T} ℓ(CB, cₜ))`

            If `aggregation` = "sum", use the sum of expertises over `T`:
            `C(CB, T) = ∑_{cₜ ∈ T} ℓ(CB, cₜ)`

            If `aggregation` = "none" or `aggregation` = `None`, return the individual expertises of `CB` over `T`:
            `C(CB, T) = [ℓ(CB, cₜ) for cₜ ∈ T]`
        :param strategy:
            If `strategy` = "MCE", use the minimum classification error loss:
            `ℓ(CB, cₜ) = - ℓmce(CB, cₜ) = - ( E(CB ∪ {(sₜ, rₜ)})) - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')})) )`.

            If `strategy` = "hinge", use the hinge competence (hinge, zero_division=0 competence = - hinge loss):
            `ℓ(CB, cₜ) = - max(0, λ + ℓmce(CB, cₜ))`.
        :param margin: 
            If `strategy` = "hinge", `margin` corresponds to the `λ` parameter.
        """
        return self.add(case_source, case_outcome).competence_case_from_cb(-1, test_cases_sources, test_cases_outcomes, strategy=strategy, margin=margin, aggregation=aggregation, **kwargs)
        
