

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
    - the energy function (energy_cb) and loss function (loss_functional_cb)
    
    Attributes
    ----------
    _X : Iterable[SourceSpaceElement]
        the situations of the cases in the CB
    _y : Iterable[OutcomeSpaceElement]
        the outcomes of the cases in the CB
    """
    
    _X: Iterable[SourceSpaceElement] = None
    _y: Iterable[OutcomeSpaceElement] = None

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
        """Computes the energy of the case base.
        
        Higher energy usually corresponds to less desirable outcomes.
        """
        raise NotImplementedError()
    def energy_case_from_cb(self, index: int, **kwargs) -> float:
        """Computes the contribution of a case to the energy."""
        return self.energy_cb(**kwargs) - self.remove(index).energy_cb(**kwargs)
    def energy_case_new(self, case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement, **kwargs) -> float:
        """Computes the contribution of a case to the energy when adding it to the CB."""
        #return self.add(case_source, case_outcome).energy_cb(**kwargs) - self.energy_cb(**kwargs)
        return self.add(case_source, case_outcome).energy_case_from_cb(-1, **kwargs)
    def energy_cases_new(self, case_source: Iterable[SourceSpaceElement], case_outcome: Iterable[OutcomeSpaceElement], **kwargs) -> Matrix:
        """Computes the contribution of one or multiple cases to the energy when adding them to the CB.
        
        This function tries all possible combinations in `case_source x case_outcome`.
        
        Returns:
            A `|case_source|` by `|case_outcome|` matrix containing the energy of all possible combinations at coordinates [source, outcome].
        """
        return np.array([[self.energy_case_new(source, outcome, **kwargs) for outcome in case_outcome] for source in case_source])
    
    @abstractmethod
    def loss_functional_cb(self, 
            test_case_source: SourceSpaceElement,
            test_case_outcome: OutcomeSpaceElement,
            **kwargs) -> float:
        """Compute the competence loss functional `ℓ(CB, cₜ)` of the case base `CB` w.r.t a test case `cₜ`.
        
        `-ℓ(CB, cₜ)` is the expertise of the case base `CB` w.r.t a test case `cₜ`.

        Higher values correspond to less desirable case/outcome combinations.

        Warning
        -------
        Since Sept. 2024, behavior has been changed to have the loss as the inverse of competence (as it should be) 
        because higher competence are more desirable, while lower losses are more desirable.

        Parameters
        ----------

        Returns
        -------
        The value of `ℓ(CB, cₜ)`
        """
        return NotImplementedError
    def loss_functional_case_from_cb(self,
            index: int, 
            test_case_source: SourceSpaceElement,
            test_case_outcome: OutcomeSpaceElement,
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            **kwargs) -> float:
        """Computes the contribution `ℓ(c, CB, cₜ)` of a case `c ∈ CB` to the loss functional w.r.t a test case `cₜ`.

        `ℓ(c, CB, cₜ) = ℓ(CB\{c}, cₜ) - ℓ(CB, cₜ)`

        Higher values correspond to less desirable case/outcome combinations.

        `-ℓ(c, CB, cₜ)` is the expertise of the case `c ∈ CB` w.r.t a test case `cₜ`.


        Warning
        -------
        Since Sept. 2024, behavior has been changed to have the loss as the inverse of competence (as it should be) 
        because higher competence are more desirable, while lower losses are more desirable.
        Also, `C(c, CB, cₜ)=ℓ(CB, cₜ) - ℓ(CB\{c}, cₜ)` was changed to `ℓ(c, CB, cₜ) = ℓ(CB\{c}, cₜ) -  ℓ(CB, cₜ)`.

        Parameters
        ----------
        strategy : "MCE" | "hinge" (default="hinge")
            If `strategy` = "MCE", use the minimum classification error loss:
            `ℓ(CB, cₜ) = ℓmce(CB, cₜ) = E(CB ∪ {(sₜ, rₜ)})) - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')}))`.

            If `strategy` = "hinge", use the hinge loss (hinge competence = - hinge loss):
            `ℓ(CB, cₜ) = max(0, λ + ℓmce(CB, cₜ))`.

            To implement another loss functional, override this method.
        margin : float 
            If `strategy` = "hinge", `margin` corresponds to the `λ` parameter.

        Returns
        -------
        The value of of the loss contribution `ℓ(c, CB, cₜ)`
        """
        return (
            self.remove(index).loss_functional_cb(test_case_source, test_case_outcome, strategy=strategy, margin=margin, **kwargs)
            -
            self.loss_functional_cb(test_case_source, test_case_outcome, strategy=strategy, margin=margin, **kwargs) 
        )
    def loss_functional_case_new(self,
            case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement,
            test_case_source: SourceSpaceElement,
            test_case_outcome: OutcomeSpaceElement,
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            **kwargs) -> float:
        """Computes the contribution `ℓ(c, CB ∪ {c}, cₜ)` of a case `c` to the loss functional w.r.t a test case `cₜ`.
        
        `ℓ(c, CB ∪ {c}, cₜ) =  ℓ(CB, cₜ) - ℓ(CB ∪ {c}, cₜ)` 

        Higher values correspond to less desirable case/outcome combinations.

        `-ℓ(c, CB ∪ {c}, cₜ)` is the expertise of the case `c` added to `CB` w.r.t a test case `cₜ`.


        Warning
        -------
        Since Sept. 2024, behavior has been changed to have the loss as the inverse of competence (as it should be) 
        because higher competence are more desirable, while lower losses are more desirable.
        Also, `C(c, CB ∪ {c}, cₜ)=ℓ(CB ∪ {c}, cₜ) - ℓ(CB, cₜ)` was changed to `ℓ(c, CB ∪ {c}, cₜ) = ℓ(CB, cₜ) -  ℓ(CB ∪ {c}, cₜ)`.

        Parameters
        ----------
        strategy : "MCE" | "hinge" (default="hinge")
            If `strategy` = "MCE", use the minimum classification error loss:
            `ℓ(CB, cₜ) = ℓmce(CB, cₜ) = E(CB ∪ {(sₜ, rₜ)})) - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')}))`.

            If `strategy` = "hinge", use the hinge loss (hinge competence = - hinge loss):
            `ℓ(CB, cₜ) = max(0, λ + ℓmce(CB, cₜ))`.

            To implement another loss functional, override this method.
        margin : float 
            If `strategy` = "hinge", `margin` corresponds to the `λ` parameter.

        Returns
        -------
        The value of of the loss contribution `ℓ(c, CB, cₜ)`
        """
        # return (
        #     self.add(case_source, case_outcome).expertise_cb(test_case_source, test_case_outcome, strategy=strategy, margin=margin, **kwargs)
        #     - self.expertise_cb(test_case_source, test_case_outcome, strategy=strategy, margin=margin, **kwargs)
        # )
        return self.add(case_source, case_outcome).loss_functional_case_from_cb(-1, test_case_source, test_case_outcome, strategy=strategy, margin=margin, **kwargs)
    def loss_cb(self,
            test_cases_sources: Iterable[SourceSpaceElement],
            test_cases_outcomes: Iterable[OutcomeSpaceElement],
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            aggregation: Literal[None, "none", "sum", "mean"] | Callable[list[float], float]="mean",
            **kwargs) -> Union[float, List[float]]:
        """Compute the loss `L(CB, T)` of the case base `CB` w.r.t a test set `T`.

        `-L(CB, T)` is the competence `C(CB, T)` of the case base `CB` w.r.t a test set `T`.
        
        Higher values correspond to less desirable case/outcome combinations.

        
        Warning
        -------
        Since Sept. 2024, behavior has been changed to have the loss as the inverse of competence (as it should be) 
        because higher competence are more desirable, while lower losses are more desirable.
        Also, `C(CB, T)` was changed to `L(CB, T)` to reflect the changes to `ℓ(CB, cₜ)`.

        Parameters
        ----------
        aggregation : Literal[None, "none", "sum", "mean"] | Callable[list[float], float] (default="mean")
            If `aggregation` = "mean", use the mean of expertises over `T`:
            `L(CB, T) = 1/|T| (∑_{cₜ ∈ T} ℓ(CB, cₜ))`

            If `aggregation` = "sum", use the sum of expertises over `T`:
            `L(CB, T) = ∑_{cₜ ∈ T} ℓ(CB, cₜ)`

            If `aggregation` = "none" or `aggregation` = `None`, return the individual expertises of `CB` over `T`:
            `L(CB, T) = [ℓ(CB, cₜ) for cₜ ∈ T]`

            If `aggregation` is callable, it is called on the list of expertises:
            `L(CB, T) = aggregation([ℓ(CB, cₜ) for cₜ ∈ T])`

        strategy : "MCE" | "hinge" (default="hinge")
            If `strategy` = "MCE", use the minimum classification error loss:
            `ℓ(CB, cₜ) = ℓmce(CB, cₜ) = E(CB ∪ {(sₜ, rₜ)})) - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')}))`.

            If `strategy` = "hinge", use the hinge loss (hinge competence = - hinge loss):
            `ℓ(CB, cₜ) = max(0, λ + ℓmce(CB, cₜ))`.

            To implement another loss functional, override this method.
        margin : float 
            If `strategy` = "hinge", `margin` corresponds to the `λ` parameter.

        Returns
        -------
        The value of of the loss `L(CB, T)`
        """
        expertises = [self.loss_functional_cb(*test_case, strategy=strategy, margin=margin, **kwargs) for test_case in zip(test_cases_sources, test_cases_outcomes)]

        if aggregation == "mean":
            aggregation = np.mean
        elif aggregation == "sum":
            aggregation = sum
        elif aggregation == "none" or aggregation is None or not callable(aggregation):
            aggregation = id # no aggregation
        return aggregation(expertises)
    def loss_case_from_cb(self,
            index: int,
            test_cases_sources: Iterable[SourceSpaceElement],
            test_cases_outcomes: Iterable[OutcomeSpaceElement],
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            aggregation: Literal[None, "none", "sum", "mean"]="mean",
            **kwargs) -> Union[float, List[float]]:
        """Compute the contribution `L(c, CB, T)` of the case `c ∈ CB` to the loss `L(CB, T)` w.r.t a test set `T`.

        Higher values correspond to less desirable case/outcome combinations.

        `-L(c, CB, T)` is the competence of the case `c ∈ CB` w.r.t a test set `T`.
        
        Warning
        -------
        Since Sept. 2024, behavior has been changed to have the loss as the inverse of competence (as it should be) 
        because higher competence are more desirable, while lower losses are more desirable.
        Also, `C(c, CB, T)` was changed to `L(c, CB, T)` to reflect the changes to `ℓ(CB, cₜ)`.

        Parameters
        ----------
        aggregation : Literal[None, "none", "sum", "mean"] | Callable[list[float], float] (default="mean")
            If `aggregation` = "mean", use the mean of expertises over `T`:
            `L(c, CB, T) = 1/|T| (∑_{cₜ ∈ T} ℓ(c, CB, cₜ))`

            If `aggregation` = "sum", use the sum of expertises over `T`:
            `L(c, CB, T) = ∑_{cₜ ∈ T} ℓ(c, CB, cₜ)`

            If `aggregation` = "none" or `aggregation` = `None`, return the individual expertises of `CB` over `T`:
            `L(c, CB, T) = [ℓ(c, CB, cₜ) for cₜ ∈ T]`

            If `aggregation` is callable, it is called on the list of expertises:
            `L(c, CB, T) = aggregation([ℓ(c, CB, cₜ) for cₜ ∈ T])`

        strategy : "MCE" | "hinge" (default="hinge")
            If `strategy` = "MCE", use the minimum classification error loss:
            `ℓ(CB, cₜ) = ℓmce(CB, cₜ) = E(CB ∪ {(sₜ, rₜ)})) - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')}))`.

            If `strategy` = "hinge", use the hinge loss (hinge competence = - hinge loss):
            `ℓ(CB, cₜ) = max(0, λ + ℓmce(CB, cₜ))`.

            To implement another loss functional, override this method.
        margin : float 
            If `strategy` = "hinge", `margin` corresponds to the `λ` parameter.

        Returns
        -------
        The value of of the loss contribution `L(c, CB, T)`
        """
        return (
            self.loss_cb(test_cases_sources, test_cases_outcomes, strategy=strategy, margin=margin, aggregation=aggregation, **kwargs)
            - self.remove(index).loss_cb(test_cases_sources, test_cases_outcomes, strategy=strategy, margin=margin, aggregation=aggregation, **kwargs)
        )
    def loss_case_new(self,
            case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement,
            test_cases_sources: Iterable[SourceSpaceElement],
            test_cases_outcomes: Iterable[OutcomeSpaceElement],
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
            aggregation: Literal[None, "none", "sum", "mean"]="mean",
            **kwargs) -> Union[float, List[float]]:
        """Compute the contribution `L(c, CB ∪ {c}, T)` of the case `c` to the loss `L(CB ∪ {c}, T)` w.r.t a test set `T`.

        Higher values correspond to less desirable case/outcome combinations.

        `-L(c, CB ∪ {c}, T)` is the competence of the case `c ∈ CB` w.r.t a test set `T`.
        
        Warning
        -------
        Since Sept. 2024, behavior has been changed to have the loss as the inverse of competence (as it should be) 
        because higher competence are more desirable, while lower losses are more desirable.
        Also, `C(c, CB, T)` was changed to `L(c, CB, T)` to reflect the changes to `ℓ(CB, cₜ)`.

        Parameters
        ----------
        aggregation : Literal[None, "none", "sum", "mean"] | Callable[list[float], float] (default="mean")
            If `aggregation` = "mean", use the mean of expertises over `T`:
            `L(c, CB ∪ {c}, T) = 1/|T| (∑_{cₜ ∈ T} ℓ(c, CB ∪ {c}, cₜ))`

            If `aggregation` = "sum", use the sum of expertises over `T`:
            `L(c, CB ∪ {c}, T) = ∑_{cₜ ∈ T} ℓ(c, CB ∪ {c}, cₜ)`

            If `aggregation` = "none" or `aggregation` = `None`, return the individual expertises of `CB` over `T`:
            `L(c, CB ∪ {c}, T) = [ℓ(c, CB ∪ {c}, cₜ) for cₜ ∈ T]`

            If `aggregation` is callable, it is called on the list of expertises:
            `L(c, CB ∪ {c}, T) = aggregation([ℓ(c, CB ∪ {c}, cₜ) for cₜ ∈ T])`

        strategy : "MCE" | "hinge" (default="hinge")
            If `strategy` = "MCE", use the minimum classification error loss:
            `ℓ(CB, cₜ) = ℓmce(CB, cₜ) = E(CB ∪ {(sₜ, rₜ)})) - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')}))`.

            If `strategy` = "hinge", use the hinge loss (hinge competence = - hinge loss):
            `ℓ(CB, cₜ) = max(0, λ + ℓmce(CB, cₜ))`.

            To implement another loss functional, override this method.
        margin : float 
            If `strategy` = "hinge", `margin` corresponds to the `λ` parameter.

        Returns
        -------
        The value of of the loss contribution `L(c, CB, T)`.
        """
        return self.add(case_source, case_outcome).loss_case_from_cb(-1, test_cases_sources, test_cases_outcomes, strategy=strategy, margin=margin, aggregation=aggregation, **kwargs)
        
