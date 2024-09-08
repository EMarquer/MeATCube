
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence, Iterable
from functools import partial, reduce
from itertools import product
import numbers
import operator
import time
import warnings

from typing import Union, List, Optional, Tuple, Literal
from typing_extensions import Self

import numpy as np
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from sklearn.base import ClassifierMixin, is_classifier#, clone
from .utils import MetaEstimatorScoreMixin as MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import indexable, check_is_fitted
from sklearn.utils.metaestimators import available_if
from sklearn.utils.parallel import delayed, Parallel
from sklearn.metrics import check_scoring
from sklearn.model_selection import train_test_split

from copy import deepcopy as clone

from .AbstractEnergyBasedClassifier import ACaseBaseEnergyClassifier, OutcomeSpaceElement, SourceSpaceElement

class CBClassificationMaintainer(MetaEstimatorMixin, ClassifierMixin):
    _required_parameters = ["estimator"]
    estimator: ACaseBaseEnergyClassifier
    refit = True

    def __init__(self,
                 estimator: ACaseBaseEnergyClassifier,
                 memorize_estimators: bool=False,
                 scoring: Union[str, callable]=None,
                 patience: int=3,
                 mode: Literal["decrement", "increment"]="decrement",
                 random_state=42,
            ):
        """
        
        Parameters
        ----------
        random_state :
            Only impacts `mode`=="increment" when explicit indices or slice are not given to initialize the process.
        scoring : str | callable (default=None)
            Strategy to evaluate the performance of the cross-validated model on the test set.

            If scoring represents a single score, one can use:
                - a single string (see The scoring parameter: defining model evaluation rules);

                - a callable (see Defining your scoring strategy from metric functions) that returns a single value.
        patience : int (default=-1) (sklearn argument)
            If `mode`=="decrement" or `mode`=="increment", `patience` is the maximum number of iterations of the 
            decremental (or incremental) process that we wait once no improvement on the score is observed.
            If `patience` steps go on (within the limits of `n_iter`) without improvements on the score, the process is 
            interrupted and the best ... is used.
        mode : Literal["decrement", "increment"] (default="decrement")
            - if `mode`=="decrement" takes the (X,y) inputs as (sources,outcomes) for the case base; then apply the decremental algorithm based on reference (X_ref, y_ref)
            - if `mode`=="increment" takes the (X,y) inputs as candidate (sources,outcomes) for the case base, picks a subset depending on `increment_init` to initializes the CB; then, apply the incremental algorithm based on reference (X_ref, y_ref)
        """
        self.estimator =estimator 
        self.memorize_estimators=memorize_estimators
        self.scoring = check_scoring(estimator, scoring)
        self.patience = patience
        self.mode = mode
        self.random_state = random_state
    
    def fit(self,
            X: Iterable[SourceSpaceElement],
            y: Iterable[OutcomeSpaceElement],
            classes: List[OutcomeSpaceElement]="infer",
            X_ref: Optional[Iterable[OutcomeSpaceElement]]=None,
            y_ref: Optional[Iterable[OutcomeSpaceElement]]=None,
            #mode: Literal["init", "decrement", "increment"]="init",
            increment_init: Optional[Union[int, Iterable[int], slice]]=None,
            n_iter: Optional[int]=None,
            #patience : int = -1,
            warm_start: bool=False,
            force_copy: bool=True,
            **loss_kwargs) -> Self:
        """The default fitting method simply takes the (X,y) inputs as (sources,outcomes) for the case base (CB).
        If modifications are to be made to the CB, only the final version is copied.
        
        Parameters
        ----------
        X : Iterable[SourceSpaceElement]
            sources of the cases to use for the CB
        y : Iterable[OutcomeSpaceElement]
            outcomes of the cases to use for the CB
        classes : List[OutcomeSpaceElement]
            list of accepted outcomes for the CB prediction
        X_ref : Iterable[SourceSpaceElement] (default=None)
            sources of the cases to use for the reference set used when `mode`=="decrement" or `mode`=="increment"
        y_ref : Iterable[OutcomeSpaceElement] (default=None)
            outcomes of the cases to use for the reference set used when `mode`=="decrement" or `mode`=="increment"
        increment_init : int or None (default=None)
            - if `increment_init` is an integer within `(0,len(X)-1)`, use `increment_init` as the number of initial cases to pick from (X,y)
            - if `increment_init` is a sequence of integers within `(0,len(X)-1)`, use `increment_init` as the list of indices to use to initialize the CB
        n_iter : int (default=-1) (sklearn argument)
            If `mode`=="decrement" or `mode`=="increment", `n_iter` is the maximum number of iterations of the 
            decremental (or incremental) process, in other words, the maximum number of cases to remove (or add).
            If `n_iter==-1`, the limit becomes the number of cases available for `mode`=="increment" and the number
            of cases in the CB minus 2 for `mode`=="decrement".
        <!--patience : int (default=-1) (sklearn argument)
            If `mode`=="decrement" or `mode`=="increment", `patience` is the maximum number of iterations of the 
            decremental (or incremental) process that we wait once no improvement on the score is observed.
            If `patience` steps go on (within the limits of `n_iter`) without improvements on the score, the process is 
            interrupted and the best ... is used. !-->
        warm_start : bool (default=False) (sklearn argument)
            reuse previous CB content; can be useful to decrement or increment with new cases an already existing CB
            If  
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
        n_features_in_ = self.estimator._check_X_y(X, y)
        if X_ref is None: raise ValueError(f"{X_ref=} but expected some data to use for current {self.mode=}")
        if y_ref is None: raise ValueError(f"{y_ref=} but expected some data to use for current {self.mode=}")
        n_features_in_ = self.estimator._check_X_y(X_ref, y_ref, n_features_in_)

        if classes == "infer":
            classes = np.unique(np.concatenate([np.unique(y), np.unique(y_ref)])).tolist()

        # init the CB
        if self.mode == "increment":
            if isinstance(increment_init, (list, slice)):
                mask = np.full(len(X),False)
                mask[increment_init] = True
                X, X_unused = X[mask], X[~mask]
                y, y_unused = y[mask], y[~mask]
            else:
                X, X_unused, y, y_unused = train_test_split(X, y, train_size=increment_init, random_state=self.random_state)
            
        if self.mode == "increment" and isinstance(increment_init, list):
            increment_init = np.random.randint(0, len(X))
        if force_copy: 
            self.estimator = clone(self.estimator)
        if not warm_start: 
            self.estimator.fit(X, y, classes, force_copy=force_copy)
        self.initial_estimator_len_ = len(self.estimator)

        # start the fitting process
        if self.memorize_estimators:
            self.estimators_ = [clone(self.estimator)]
            self.best_estimator_ = self.estimators_[0]
        else:
            self.best_estimator_ = clone(self.estimator)
        self.scores_ = [self.estimator.score(X_ref, y_ref)]
        self.best_score_ = self.scores_[0]
        self.best_index_ = 0
        check_is_fitted(self.estimator)

        if n_iter is None:
            if self.mode == "decrement":
                n_iter = len(self.estimator._X) - 1
            elif self.mode == "increment":
                n_iter = len(X_unused)
        else:
            if self.mode == "decrement":
                n_iter = min(n_iter, len(self.estimator._X) - 1)
            elif self.mode == "increment":
                n_iter = min(n_iter, len(X_unused))

        for i in range(n_iter):
            # update the estimator
            if self.mode == "decrement":
                self.decrement(X_ref=X_ref, y_ref=y_ref, inplace=True, **loss_kwargs)
            elif self.mode == "increment":
                self.increment(X_unused, y_unused, X_ref, y_ref, inplace=True, **loss_kwargs)

            # evaluate the latest model and update best model
            self.scores_.append(self.score(X_ref, y_ref))
            if self.memorize_estimators:
                self.estimators_.append(self.estimator)
            if self.scores_[-1] > self.best_score_: # update best model
                self.best_score_ = self.scores_[-1]
                if self.memorize_estimators:
                    self.best_estimator_ = self.estimators_[-1]
                else:
                    self.best_estimator_ = self.estimator
                self.best_index_ = i+1  

            # if enough iterations have been performed, check if the performance has improved in the last `patience` 
            # iterations` 
            if len(self.scores_) >= self.patience and max(self.scores_[-self.patience-1:-1]) > self.scores_[-1]:
                break
        

        self.estimator = self.best_estimator_
        check_is_fitted(self.estimator)
        return self#.estimator#, self.best_score_

    def score(self, X, y, sample_weight: None=None) -> float:
        check_is_fitted(self.estimator)
        return self.estimator.score(X=X, y=y)

    def increment(self,
            X: Iterable[SourceSpaceElement],
            y: Iterable[OutcomeSpaceElement],
            X_ref: Optional[Iterable[OutcomeSpaceElement]]=None,
            y_ref: Optional[Iterable[OutcomeSpaceElement]]=None,
            inplace: bool=False,
            
            **loss_kwargs) -> int:
        """Adds a case to the CB by picking the most competent.
        
        Takes the (X,y) inputs as candidate (sources,outcomes) for the case base. Then, applies the incremental 
        algorithm using reference (X_ref, y_ref) to determine the cases increasing the competence by the largest margin,
        and adding them to the CB.

        If modifications are to be made to the CB, only the final version is copied.

        Parameters
        ----------
        X : Iterable[SourceSpaceElement]
            sources of the cases that can be added to the CB
        y : Iterable[OutcomeSpaceElement]
            outcomes of the cases that can be added to the CB
        X_ref : Iterable[SourceSpaceElement] (default=None)
            sources of the cases to use for the reference set
        y_ref : Iterable[OutcomeSpaceElement] (default=None)
            outcomes of the cases to use for the reference set
        inplace : bool (default=False)
            If True, self.estimator is replaced by the new estimator.

        Returns
        -------
        updated_classifier : ACaseBaseEnergyClassifier
            The estimator without the removed case.
        index : int
            The index of the added case.
        increment_scores:
            The scores that led to the addition decision.
        """
        increment_scores = self.estimator.increment_scores(X_candidate=X, y_candidate=y, X_ref=X_ref, y_ref=y_ref, **loss_kwargs)
        index = increment_scores.argmin()
        estimator = self.estimator.add(X[index], y[index])
        if inplace:
            self.estimator = estimator
        
        return estimator, index, increment_scores
    def decrement(self,
            X_ref: Optional[Iterable[OutcomeSpaceElement]]=None,
            y_ref: Optional[Iterable[OutcomeSpaceElement]]=None,
            inplace: bool=False,
            **loss_kwargs) -> Tuple[ACaseBaseEnergyClassifier, int]:
        """Removes a case from the CB by picking the least competent.
        
        Applies the decremental algorithm based on reference (X_ref, y_ref) to determine the cases decreasing the 
        competence by the largest margin, and removing them.
        
        If modifications are to be made to the CB, only the final version is copied.

        
        Parameters
        ----------
        X_ref : Iterable[SourceSpaceElement] (default=None)
            sources of the cases to use for the reference set
        y_ref : Iterable[OutcomeSpaceElement] (default=None)
            outcomes of the cases to use for the reference set
        inplace : bool (default=False)
            If True, self.estimator is replaced by the new estimator.

        Returns
        -------
        updated_classifier : ACaseBaseEnergyClassifier
            The estimator without the removed case.
        index : int
            The index of the removed case.
        decrement_scores:
            The scores that led to the removal decision.
        """
        decrement_scores = self.estimator.decrement_scores(X_ref=X_ref, y_ref=y_ref, **loss_kwargs)
        index = decrement_scores.argmax()
        estimator = self.estimator.remove(index)
        if inplace:
            self.estimator = estimator
        
        return estimator, index, decrement_scores
    
