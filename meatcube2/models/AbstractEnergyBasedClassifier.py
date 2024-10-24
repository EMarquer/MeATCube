"""Refer to https://scikit-learn.org/stable/developers/develop.html for dev instructions.

"The object's __init__ method might accept constants as arguments that determine the estimator's behavior (like the C 
constant in SVMs). It should not, however, take the actual training data as an argument, as this is left to the fit() 
method. The arguments accepted by __init__ should all be keyword arguments with a default value. In other words, a user
should be able to instantiate an estimator without passing any arguments to it.

[...]

There should be no logic, not even input validation, and the parameters should not be changed. The corresponding logic
should be put where the parameters are used, typically in [the fit() method]."

"Attributes that have been estimated from the data must always have a name ending with trailing underscore, for example 
the coefficients of some regression estimator would be stored in a coef_ attribute after fit has been called.

The estimated attributes are expected to be overridden when you call fit a second time."



"""
from __future__ import annotations # for self-referring type hints
from typing import Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List
from typing_extensions import Self
from collections.abc import Sequence
from abc import abstractmethod, ABC, ABCMeta

import numpy as np
import torch
from .AbstractEnergyBasedPredictor import ACaseBaseEnergyPredictor, SourceSpaceElement, OutcomeSpaceElement

from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.special import softmax
from ..utils import to_numpy_array

class ACaseBaseEnergyClassifier(BaseEstimator, ClassifierMixin, ACaseBaseEnergyPredictor):
    """
    
    Attributes
    ----------
    classes_ : List[OutcomeSpaceElement]
        the list of possible classes in the CB
    n_features_in_: int
        number of features expected in the situation space

    Inherited attributes
    ----------
    _X : Iterable[SourceSpaceElement]
        the situations of the cases in the CB
    _y : Iterable[OutcomeSpaceElement]
        the outcomes of the cases in the CB
    """
    
    classes_: List[OutcomeSpaceElement] = None

    def fit(self,
                 X: Iterable[SourceSpaceElement], 
                 y: Iterable[OutcomeSpaceElement],
                 classes: List[OutcomeSpaceElement] | Literal["infer"]="infer", 
                 force_copy=True) -> Self:
        """Initializes attributes depending on the CB.
        
        Parameters
        ----------
        X : Iterable[SourceSpaceElement]
            sources of the cases to use for the CB
        y : Iterable[OutcomeSpaceElement]
            outcomes of the cases to use for the CB
        classes : List[OutcomeSpaceElement]
            list of accepted outcomes for the CB prediction
        force_copy : bool (default=True)
            If True, the values in (X,y) are copied to a new container, ensuring that no change made to the original 
            (X,y) arrays (or to this object) will impact this object (respectively, the original (X,y) arrays).
            Set to False to minimize RAM usage, but be careful of the risks.
        """
        # init checks
        self.n_features_in_ = self._check_X_y(X, y)

        if force_copy: 
            X=np.copy(X)
            y=np.copy(y)
        self._X = to_numpy_array(X)
        self._y = to_numpy_array(y)
        if classes == "infer":
            classes = np.unique(self._y).tolist()
        self.classes_ = classes

        return self

    def loss_functional_cb(self, 
            test_case_source: SourceSpaceElement,
            test_case_outcome: OutcomeSpaceElement,
            strategy: Literal["MCE", "hinge"]="hinge",
            margin: float=0.1,
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
        The value of `ℓ(CB, cₜ)`
        """
        if strategy == "MCE":
            # find `rₜ' ≠ rₜ`: candidate_classes = {rₜ' ∈ self.classes_ | rₜ' ≠ rₜ}
            if self.classes_.__contains__(test_case_outcome):
                outcome_index = self.classes_.index(test_case_outcome)
                candidate_classes = self.classes_[:outcome_index] + self.classes_[outcome_index+1:]
            else:
                candidate_classes = self.classes_

            return (
                self.energy_case_new(test_case_source, test_case_outcome, **kwargs) # E(CB ∪ {(sₜ, rₜ)}
                - self.predict_one(test_case_source, candidate_classes=candidate_classes, return_energies=True, **kwargs)[1][0] # min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')})
            )
        elif strategy == "hinge":
            return max(0, margin + self.loss_functional_cb(test_case_source, test_case_outcome, strategy="MCE", **kwargs))
        else: 
            raise ValueError(f"Unexpected competence/expertise strategy: {strategy}")

    def predict(self, X: Iterable[SourceSpaceElement], candidate_classes=None) -> Iterable[OutcomeSpaceElement]:
        # Check if fit has been called
        check_is_fitted(self)
        check_array(X,  accept_sparse=True, dtype=None, ensure_2d=False, allow_nd=False)
        
        # use self.classes_ by default
        if candidate_classes is None:
            candidate_classes = self.classes_
        
        return self.predict_multiple(X)
    
    def predict_one(self, X: SourceSpaceElement, candidate_classes=None, return_id=False, return_energies=False, **kwargs) -> Union[
            OutcomeSpaceElement, Tuple[OutcomeSpaceElement, int], Tuple[OutcomeSpaceElement, Iterable[float]], Tuple[OutcomeSpaceElement, int, Iterable[float]]]:
        """Compute the energy of a new situation combined with each of `candidate_classes`.
        
        Parameters
        ----------
        X : 
            The source of the case to predict.
        candidate_classes :
            If provided, use as a list of possible classes, otherwise use `self.classes_`.
        return_id:
            If True, returns the class index corresponding to each class.
        return_energies:
            If True, returns the energies corresponding to each class.

        Returns
        -------
        outcome(, index)(, energy):
            Depends on the values of `return_id` and `return_energy`
            - if `return_id==False` and `return_energy==False`, returns the label only
            - if `return_id==True`  and `return_energy==False`, returns the label and the corresponding index
            - if `return_id==False` and `return_energy==True`, returns the label and the corresponding energy
            - if `return_id==True`  and `return_energy==True`, returns the label, the corresponding index, and the corresponding energy
        """
        # Check if fit has been called
        check_is_fitted(self)
        check_array(X,  accept_sparse=True, dtype=None, ensure_2d=False, allow_nd=False)
        
        # use self.classes_ by default
        if candidate_classes is None:
            candidate_classes = self.classes_
        
        energies = self.energy_cases_new([X], candidate_classes, **kwargs)[0]
        if isinstance(energies, torch.Tensor): 
            min_index = energies.argmin().cpu().item() # failsafe if energy_cases_new returns a tensor
        else: 
            min_index = np.argmin(energies).item()

        if return_id and return_energies:
            return candidate_classes[min_index], min_index, energies#[min_index]
        elif return_id:
            return candidate_classes[min_index], min_index
        elif return_energies:
            return candidate_classes[min_index], energies#[min_index]
        else:
            return candidate_classes[min_index]
    def predict_multiple(self, X: Iterable[SourceSpaceElement], candidate_classes=None, return_id=False, return_energies=False, **kwargs) -> Union[
            Iterable[OutcomeSpaceElement], Tuple[Iterable[OutcomeSpaceElement], Iterable[int]], Tuple[Iterable[OutcomeSpaceElement], Iterable[Iterable[float]]], Tuple[Iterable[OutcomeSpaceElement], Iterable[int], Iterable[Iterable[float]]]]:
        """Compute the energy of new situations combined with each of `candidate_classes`.
        
        Parameters
        ----------
        X : 
            The sources of the cases to predict.
        candidate_classes :
            If provided, use as a list of possible classes, otherwise use `self.classes_`.
        return_id:
            If True, returns the class index corresponding to each class.
        return_energies:
            If True, returns the energies corresponding to each class.

        Returns
        -------
        outcome(, index)(, energy): List[OutcomeSpaceElement] (, List[float])(, List[Iterable])
            Depends on the values of `return_id` and `return_energy`
            - if `return_id==False` and `return_energy==False`, returns the label only
            - if `return_id==True`  and `return_energy==False`, returns the label and the corresponding index
            - if `return_id==False` and `return_energy==True`, returns the label and the corresponding energy
            - if `return_id==True`  and `return_energy==True`, returns the label, the corresponding index, and the corresponding energy
        """
        # Check if fit has been called
        check_is_fitted(self)
        check_array(X,  accept_sparse=True, dtype=None, ensure_2d=False, allow_nd=False)
        
        # use self.classes_ by default
        if candidate_classes is None:
            candidate_classes = self.classes_
        
        # compute the energies
        energies = self.energy_cases_new(X, candidate_classes, as_tensor=True)
        
        # get class/outcome indices minimizing the energy
        if isinstance(energies, torch.Tensor): 
            min_index = energies.argmin(dim=-1).cpu().numpy() # failsafe if energy_cases_new returns a tensor
            energies = energies.cpu().numpy()
        else: 
            min_index = np.argmin(energies, axis=-1)

        # transform class/outcome indices into class/outcome predictions
        predictions = np.vectorize(candidate_classes.__getitem__)(min_index)
        
        if return_id and return_energies:
            return predictions, min_index, energies#[min_index]
        elif return_id:
            return predictions, min_index
        elif return_energies:
            return predictions, energies#[min_index]
        else:
            return predictions
        # except Exception as e:
        #     predictions = [self.predict_one(x, candidate_classes=candidate_classes, return_id=return_id, return_energies=return_energies, **kwargs) for x in X]
        #     if return_id or return_energies:
        #         return (list(sequence) for sequence in zip(*predictions)) # transform a list of tuples into a tuple of lists
        #     else:
        #         return predictions


    def predict_proba(self, X: Iterable[SourceSpaceElement], candidate_classes=None, temperature=1) -> Iterable[Iterable[float]]:
        """Returns a tensor with, for each possible outcome, the confidence (as defined below) in the prediction for `sources`.

        The confidence for a class y is the result of the scaled exponential (or scaled softmax):
        confidence(X, y) = exp(-E(X, y) / temperature) / ∑_{y' ∈ classes_}(np.exp(-E(X, y') / temperature))

        Parameters
        ----------
        X : Iterable[SourceSpaceElement]
            _description_
        candidate_classes : _type_, optional
            _description_, by default None
        temperature : int, optional
            _description_, by default 1

        Returns
        -------
        Iterable[Iterable[float]]
            _description_
        """
        # Check if fit has been called
        check_is_fitted(self)
        check_array(X,  accept_sparse=True, dtype=None, ensure_2d=False, allow_nd=False)
        
        # check temperature value
        assert temperature >= 1, f"Prediction temperature is {temperature} but expected temperature >= 1"

        # use self.classes_ by default
        if candidate_classes is None:
            candidate_classes = self.classes_

        _, energies = self.predict_multiple(X, return_energies=True)
        energies = np.array(energies, dtype=float)
        probas = softmax(-energies / temperature, axis=-1)
        
        return probas
    
    def _check_X_y(self, 
                    X: Iterable[SourceSpaceElement], 
                    y: Iterable[OutcomeSpaceElement], 
                    n_features_in: Optional[int]=None) -> Optional[int]:
        """_summary_

        Parameters
        ----------
        X : Iterable[SourceSpaceElement]
            _description_
        y : Iterable[OutcomeSpaceElement]
            _description_
        n_features_in : Optional[int], optional
            _description_, by default None

        Returns
        -------
        n_features_in : Optional[int]
            None if no second dimension is present in `X`, otherwise dimension of the second dimension of `X`

        Raises
        ------
        ValueError
            If the first dimension of `X` and `y` do not match.
        ValueError
            If `n_features_in` specified but no second dimension of `X` is available.
        ValueError
            If `n_features_in` specified but the second dimension of `X` is not large enough.
        TypeError
            If `X` and `y` do not have a matching type among [(list,list), (np.ndarray,np.ndarray), (torch.Tensor,torch.Tensor)].
            
        """
        try:
            check_X_y(X,y,  accept_sparse=True, dtype=None, ensure_2d=False, allow_nd=False)
        except ValueError:
            print(X,y)
        if isinstance(X, (str)): raise TypeError
        if isinstance(y, (str)): raise TypeError
        # if isinstance(X, (np.ndarray)) and X.dtype in [object]: raise TypeError
        # if isinstance(y, (np.ndarray)) and y.dtype in [object]: raise TypeError
        if isinstance(X, (list, tuple)) and isinstance(y, (list, tuple)):
            if len(X) != len(y): raise ValueError(f"len(X) ({len(X)}) != len(y) ({len(y)})")
            if len(X) < 1: raise ValueError(f"len(X) ({len(X)}) < 0")

            # count how many features are present in the matrix, and if necessary compare with specified feature number
            if n_features_in is not None:
                if isinstance(X[0], (list, tuple)):
                    assert n_features_in == len(X)
                elif isinstance(X[0], np.ndarray):
                    assert n_features_in == X.shape[0]
                elif isinstance(X[0], torch.Tensor):
                    assert n_features_in == X.size(0)
                else: raise ValueError(f"specified {n_features_in=} but feature checks are not supported for {type(X[0])=}")
            else:
                if isinstance(X[0], (list, tuple)):
                    n_features_in = len(X)
                elif isinstance(X[0], np.ndarray):
                    n_features_in = X.shape[0]
                elif isinstance(X[0], torch.Tensor):
                    n_features_in = X.size(0)
                else:
                    n_features_in = None

        elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            if X.shape[0] != y.shape[0]: raise ValueError(f"X.shape[0] ({X.shape[0]}) != y.shape[0] ({y.shape[0]})")
            if X.shape[0] < 1: raise ValueError(f"X.shape[0] ({X.shape[0]}) < 0")

             # count how many features are present in the matrix, and if necessary compare with specified feature number
            if len(X.shape)>1:
                if n_features_in is not None:
                    assert n_features_in == X.shape[1]
                else:
                    n_features_in = X.shape[1]
            elif n_features_in is not None: raise ValueError(f"specified {n_features_in=} but feature checks are not supported for X of dimension {len(X.shape)=}")
        elif isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):
            if X.size(0) != y.size(0): raise ValueError(f"X.size(0) ({X.size(0)}) != y.size(0) ({y.size(0)})")
            if X.size(0) < 1: raise ValueError(f"X.size(0) ({X.size(0)}) < 0")

            # count how many features are present in the matrix
            if X.dim()>1:
                if n_features_in is not None:
                    assert n_features_in == X.size(1)
                else:
                    n_features_in = X.size(1)
            elif n_features_in is not None: raise ValueError(f"specified {n_features_in=} but feature checks are not supported for X of dimension {X.dim()=}")
        else: raise TypeError(f"Unknown combination of (X,y), expected one of [(list,list), (np.ndarray,np.ndarray), (torch.Tensor,torch.Tensor)], got ({type(X)},{type(y)})")

        return n_features_in

    #@abstractmethod
    def decrement_scores(self, X_ref, y_ref, **loss_kwargs) -> np.ndarray[float]:
        """Return the loss contribution of each case in the CB w.r.t each case in the reference set (X_ref,y_ref).
        
        When performing decremental maintenance (removing cases), these scores are used to select the case(s) to remove.

        Higher values correspond to less desirable cases.

        Returns
        -------
        loss_contributions : for each case in the CB, returns the contributions of the case to the loss with regards to the reference set (X_ref,y_ref)
        """
        loss_contributions = np.array([self.loss_case_from_cb(i, X_ref, y_ref, **loss_kwargs) for i in range(len(self))])
        return loss_contributions
    #@abstractmethod
    def increment_scores(self, X_candidate, y_candidate, X_ref, y_ref, **loss_kwargs) -> np.ndarray[float]:
        """Return the loss contribution of each case in the candidate set added to the CB w.r.t each case in the 
        reference set (X_ref,y_ref).
        
        When performing incremental maintenance (adding cases), these scores are used to select the case(s) to add.

        Higher values correspond to less desirable cases.

        Returns
        -------
        loss_contributions : for each case in the candidate set (X_candidate, y_candidate), returns the contributions of the case to the loss with regards to the reference set (X_ref,y_ref)
        """
        loss_contributions = np.array([self.loss_case_new(X_, y_, X_ref, y_ref, **loss_kwargs) for (X_, y_) in zip(X_candidate, y_candidate)])
        return loss_contributions
        
    def __getitem__(self, index) -> Tuple:
        return self._X[index], self._y[index]
    def __len__(self) -> int:
        return max(len(self._X), len(self._y))
