
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence, Iterable
from functools import partial, reduce
from itertools import product
import numbers
import operator
import time
import warnings

from typing import Union, List, Optional, Tuple, Literal, Any
from typing_extensions import Self

import numpy as np
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from sklearn.base import ClassifierMixin, is_classifier#, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import indexable, check_is_fitted
from sklearn.utils.metaestimators import available_if
from sklearn.utils.parallel import delayed, Parallel
from sklearn.metrics import check_scoring
from sklearn.model_selection import train_test_split

from copy import deepcopy as clone

from ..models.AbstractEnergyBasedClassifier import ACaseBaseEnergyClassifier, OutcomeSpaceElement, SourceSpaceElement
from ..metrics import clf_prediction_summary
from ..utils import catchtime
from ..utils import MetaEstimatorScoreMixin as MetaEstimatorMixin
from .cb_mainainer import CBClassificationMaintainerBase

class EnergyBasedMaintainer(CBClassificationMaintainerBase):
    """
    """
    
    def __init__(self,
                 estimator: ACaseBaseEnergyClassifier,
                 memorize_estimators: bool=False,
                 scoring: Union[str, callable]=clf_prediction_summary,
                 refit: str="accuracy",
                 patience: int=-1,
                 mode: Literal["decrement", "increment"]="decrement",
                 random_state=42,
            ):
        """
        
        Parameters
        ----------
        mode : Literal["decrement", "increment"] (default="decrement")
            - if `mode`=="decrement" takes the (X,y) inputs as (sources,outcomes) for the case base; then apply the decremental algorithm based on reference (X_ref, y_ref)
            - if `mode`=="increment" takes the (X,y) inputs as candidate (sources,outcomes) for the case base, picks a subset depending on `increment_init` to initializes the CB; then, apply the incremental algorithm based on reference (X_ref, y_ref)
        """
        super().__init__(estimator, memorize_estimators, scoring, refit, patience, random_state)
        self.mode = mode
    
    def before_fit_loop(
            self,
            X: Iterable[SourceSpaceElement],
            y: Iterable[OutcomeSpaceElement],
            X_ref: Iterable[OutcomeSpaceElement],
            y_ref: Iterable[OutcomeSpaceElement],
            classes: List[OutcomeSpaceElement]="infer",
            increment_init: Optional[Union[int, Iterable[int], slice]]=None,
            n_iter: Optional[int]=None,
            #patience : int = -1,
            warm_start: bool=False,
            force_copy: bool=True,
            loss_kwargs=dict(),
            **kwargs
        ):
        """Called before running the maintenance loop.
        All the preparations are done here.
        
        When implementing subclass and overwriting this method, it is highly recommended to call it as super().before_fit_loop(...) ."""

        # init the CB
        if self.mode == "increment":
            if isinstance(increment_init, (list, slice)):
                mask = np.full(len(X),False)
                mask[increment_init] = True
                X, X_unused = X[mask], X[~mask]
                y, y_unused = y[mask], y[~mask]
            else:
                X, X_unused, y, y_unused = train_test_split(X, y, train_size=increment_init, random_state=self.random_state)
        else: 
            X_unused = y_unused = None
        if self.mode == "decrement":
            self.kept_cases_ = list(range(len(X)))
            self.removed_cases_ = []
            
        if self.mode == "increment" and isinstance(increment_init, list):
            increment_init = np.random.randint(0, len(X))
        
        super().before_fit_loop(X, y, X_ref, y_ref, classes=classes, n_iter=n_iter, warm_start=warm_start, force_copy=force_copy)
        
        # find the maximum number of iterations
        if self.n_iter_ is None:
            if self.mode == "decrement":
                self.n_iter_ = len(self.estimator_._X) - 1
            elif self.mode == "increment":
                self.n_iter_ = len(X_unused)
        else:
            if self.mode == "decrement":
                self.n_iter_ = min(self.n_iter_, len(self.estimator_._X) - 1)
            elif self.mode == "increment":
                self.n_iter_ = min(self.n_iter_, len(X_unused))


        self.X_unused_ = X_unused
        self.y_unused_ = y_unused
        self.increment_init_ = increment_init
        self.loss_kwargs_ = loss_kwargs
    
    def after_fit_loop(self):
        """Executed after the fit loop. Use to clean up or refit if necessary, for instance."""
        del self.X_unused_
        del self.y_unused_
        del self.increment_init_
        del self.loss_kwargs_
        super().after_fit_loop()

    def update(self) -> None:
        if self.mode == "decrement":
            self.decrement()
        elif self.mode == "increment":
            self.increment()

    def increment(self) -> None:
        """Adds a case to the CB by picking the most suitable.
        
        By default, uses the definition of competence using the energy to determine the best case to add.
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
            If True, self.estimator_ is replaced by the new estimator.

        Returns
        -------
        updated_classifier : ACaseBaseEnergyClassifier
            The estimator without the removed case.
        index : int
            The index of the added case.
        """
        increment_scores = self.estimator_.increment_scores(X_candidate=self.X_unused_, y_candidate=self.y_unused_, X_ref=self.X_ref_, y_ref=self.y_ref_, **self.loss_kwargs_)
        index = increment_scores.argmax()
        estimator = self.estimator_.add(self.X_unused_[index], self.y_unused_[index])
        self.estimator_ = estimator
    def decrement(self) -> None:
        """Removes a case from the CB by picking the least suitable.
        
        By default, uses the definition of competence using the energy to determine the best case to remove.
        Applies the decremental algorithm based on reference (X_ref, y_ref) to determine the cases decreasing the 
        competence by the largest margin, and removing them.
        
        If modifications are to be made to the CB, only the final version is copied.

        
        Parameters
        ----------
        X_ref : Iterable[SourceSpaceElement] (default=None)
            sources of the cases to use for the reference set
        y_ref : Iterable[OutcomeSpaceElement] (default=None)
            outcomes of the cases to use for the reference set
        """
        decrement_scores = self.estimator_.decrement_scores(X_ref=self.X_ref_, y_ref=self.y_ref_, **self.loss_kwargs_)
        index = np.nanargmin(decrement_scores)
        initial_index = self.kept_cases_.pop(index)
        self.removed_cases_.append(initial_index)
        estimator = self.estimator_.remove(index)
        self.estimator_ = estimator
        

class EnergyCompress(EnergyBasedMaintainer):
    def __init__(self,
                 estimator,
                 memorize_estimators = False, 
                 scoring = clf_prediction_summary, 
                 refit = "accuracy", 
                 patience = -1, 
                 random_state=42):
        super().__init__(estimator=estimator,
                         memorize_estimators=memorize_estimators,
                         scoring=scoring, 
                         refit=refit, 
                         patience=patience, 
                         mode="decrement", 
                         random_state=random_state)