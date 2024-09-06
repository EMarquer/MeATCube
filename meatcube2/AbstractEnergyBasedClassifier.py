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

from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class ACaseBaseEnergyClassifier(BaseEstimator, ClassifierMixin, ACaseBaseEnergyPredictor):
    
    
    @abstractmethod
    def fit_input(self,
                 X: Iterable[SourceSpaceElement], y: Iterable[OutcomeSpaceElement],
                 potential_outcomes: List[OutcomeSpaceElement]="auto",
                 device = "auto") :
        """Initializes attributes depending on the CB."""
        raise NotImplementedError

    def fit(self,
            X: Iterable[SourceSpaceElement], y: Iterable[OutcomeSpaceElement],
            X_ref: Optional[Iterable[OutcomeSpaceElement]]=None, y_ref: Optional[Iterable[OutcomeSpaceElement]]=None,
            mode: Optional[str]="init", increment_init: Optional[Union[int, Iterable[int]]]=None,
            n_iter: Optional[int]=None,
            warm_start: bool=False,
            force_copy: bool=True) -> Self:
        """The default fitting method simply takes the (X,y) inputs as (sources,outcomes) for the case base (CB).
        If modifications are to be made to the CB, only the final version is copied.
        
        Parameters
        ----------
        mode : string
            TODO: add modes
                - if `mode`=="init" (default): takes the (X,y) inputs as (sources,outcomes) for the case base, i.e., initializes the CB with the input
                - if `mode`=="decrement" takes the (X,y) inputs as (sources,outcomes) for the case base; then apply the decremental algorithm based on reference (X_ref, y_ref)
                - if `mode`=="increment" takes the (X,y) inputs as candidate (sources,outcomes) for the case base, picks a subset depending on `increment_init` to initializes the CB; then, apply the incremental algorithm based on reference (X_ref, y_ref)
        increment_init : int or None (default=None)
            - if `increment_init` is an integer within `(0,len(X)-1)`, use `increment_init` as the number of initial cases to pick from (X,y)
            - if `increment_init` is a sequence of integers within `(0,len(X)-1)`, use `increment_init` as the list of indices to use to initialize the CB
        n_iter : bool (default=None) (sklearn argument)
            if `mode`=="decrement" or `mode`=="increment", `n_iter` is the maximum number of iterations of the 
            decremental (or incremental) process, in other words, the maximum number of cases to remove (or add)
        warm_start : bool (default=False) (sklearn argument)
            reuse previous CB content; can be useful to decrement or increment with new cases an already existing CB 
        force_copy : bool (default=True)
            If True, the values in (X,y) are copied to a new container, ensuring that no change made to the original 
            (X,y) arrays (or to this object) will impact this object (respectively, the original (X,y) arrays).
            Set to False to minimize RAM usage, but be careful of the risks.

        Attributes
        -------
        n_features_in_ : int (sklearn argument)
            number of features that the estimator expects for subsequent calls to predict

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        # init checks
        n_features_in_ = self.init_checks(X, y)
        if mode in ("decrement", 'increment'):
            self.init_checks(X_ref, y_ref, n_features_in_)

        # init the CB
        self.fit_input(X, y)
        self.n_features_in_ = n_features_in_

        # if required, start the fitting process

        return self
    
    def predict(self, X: Iterable[SourceSpaceElement]) -> Iterable[OutcomeSpaceElement]:

        # Check if fit has been called

        check_is_fitted(self)
        check_array(X,  accept_sparse=True, dtype=None, ensure_2d=False, allow_nd=False)
        
        return self.predict_multiple(X)
    
    def init_checks(self, 
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
        check_X_y(X,y,  accept_sparse=True, dtype=None, ensure_2d=False, allow_nd=False)
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

    @abstractmethod
    def decrement_scores(self, X_ref, y_ref, **competence_kwargs) -> List[float]:
        """Return the competence of each case in the CB w.r.t each case in the reference set (X_ref,y_ref).
        
        When performing decremental maintenance (removing cases), these scores are used to select the case(s) to remove.

        Returns
        -------
        expertises : for each case in the CB, returns the expertise of the case with regards to the reference set (X_ref,y_ref)
        """
        expertises = np.array([self.competence_case_from_cb(i, X_ref, y_ref, **competence_kwargs) for i in range(len(self))])
        return expertises
    @abstractmethod
    def increment_scores(self, X_candidate, y_candidate, X_ref, y_ref, **competence_kwargs) -> List[float]:
        """Return the competence of each case in the candidate set added to the CB w.r.t each case in the reference set 
        (X_ref,y_ref).
        
        When performing incremental maintenance (adding cases), these scores are used to select the case(s) to add.

        Returns
        -------
        expertises : for each case in the candidate set (X_candidate, y_candidate), returns the expertise of the case with regards to the reference set (X_ref,y_ref)
        """
        expertises = np.array([self.competence_case_new(X_, y_, X_ref, y_ref, **competence_kwargs) for (X_, y_) in zip(X_candidate, y_candidate)])
        return expertises
        
        

# from sklearn.utils.estimator_checks import check_estimator
# class EuclideanCBClassifier(ACaseBaseEnergyClassifier):

#     pass

# assert check_estimator(ACaseBaseEnergyClassifier())

# class ACaseBaseEnergyRegressor(RegressorMixin, ACaseBaseEnergyPredictor):
#     pass