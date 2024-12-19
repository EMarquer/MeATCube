
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence, Iterable
from functools import partial, reduce
from itertools import product
import numbers
import operator
import time
import warnings

from typing import Union, List, Optional, Tuple, Literal, Any, Callable
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

class CBClassificationMaintainerBase(MetaEstimatorMixin, ClassifierMixin):
    """
    To summarize the process:
    1. before_fit_loop(...): all the initialization steps
    2. for iteration in range(self.n_iter_):
        1. update the CB with self.update(), required variables are set in before_fit_loop(...) and updated in self.update()
        2. compute the score(s) with self.get_score_to_watch(self)
        3. if self.stopping_criterion()[0], stop the process
    3. after_fit_loop(...): all the cleanup steps
    """
    _required_parameters = ["estimator"]
    estimator: ACaseBaseEnergyClassifier
    scoring: Callable
    refit = None

    def __init__(self,
                 estimator: ACaseBaseEnergyClassifier,
                 memorize_estimators: bool=False,
                 scoring: Union[str, callable]=clf_prediction_summary,
                 refit: str="accuracy",
                 patience: int=-1,
                 random_state=42,
            ):
        """
        
        Parameters
        ----------
        random_state :
            Only impacts `mode`=="increment" when explicit indices or slice are not given to initialize the process.
        scoring : str | callable (default=metrics.clf_prediction_summary) (sklearn argument)
            Strategy to evaluate the performance of the cross-validated model on the test set.

            If scoring represents a single score, one can use:
                - a single string (see The scoring parameter: defining model evaluation rules);

                - a callable (see Defining your scoring strategy from metric functions) that returns a single value.
        refit : callable, default=score (sklearn argument)
            By analogy with the refit parameter of GridSearchCV.

            For multiple metric evaluation, this needs to be a str denoting the scorer that would be used to find the 
            best parameters for the estimator at the end.
        patience : int (default=-1) (sklearn argument)
            the maximum number of iterations of the process that we wait once no improvement on the score is observed.
            If `patience` < 0, steps go on within the limits of `n_iter` and the best estimator is used.
        """
        self.estimator = estimator 
        self.memorize_estimators=memorize_estimators
        self.scoring = check_scoring(estimator, scoring)
        self.refit = refit
        self.patience = patience
        self.random_state = random_state
    
    def before_fit_loop(
            self,
            X: Iterable[SourceSpaceElement],
            y: Iterable[OutcomeSpaceElement],
            X_ref: Iterable[OutcomeSpaceElement],
            y_ref: Iterable[OutcomeSpaceElement],
            classes: List[OutcomeSpaceElement]="infer",
            n_iter: Optional[int]=None,
            warm_start: bool=False,
            force_copy: bool=True,
            fit_kwargs=dict(),
            **kwargs,
        ):
        """Called before running the maintenance loop.
        All the preparations are done here.
        
        When implementing subclass and overwriting this method, it is highly recommended to call it as super().before_fit_loop(...) ."""
        if X_ref is None and y_ref is None:
            X_ref, y_ref = X, y
        
        
        if classes == "infer":
            classes = np.unique(np.concatenate([np.unique(y), np.unique(y_ref)])).tolist()
        
        if force_copy: 
            self.estimator_ = clone(self.estimator)
        else:
            self.estimator_ = (self.estimator)
        
        if not warm_start: 
            self.estimator_.fit(X, y, classes, force_copy=force_copy, **fit_kwargs)
        self.initial_estimator_len_ = len(self.estimator_)

        # init checks
        n_features_in_ = self.estimator_._check_X_y(X, y)
        n_features_in_ = self.estimator_._check_X_y(X_ref, y_ref, n_features_in_)

        self.X_ref_ = X_ref
        self.y_ref_ = y_ref
        self.X_ = X
        self.y_ = y
        self.n_iter_ = n_iter
        self.fit_kwargs_ = fit_kwargs
        self.force_copy_ = force_copy
    
    def after_fit_loop(self):
        """Executed after the fit loop. Use to clean up or refit if necessary, for instance."""
        del self.X_ref_
        del self.y_ref_
        del self.X_
        del self.y_
        del self.n_iter_
        del self.fit_kwargs_
        del self.force_copy_

    def get_score_to_watch(self) -> float:
        with catchtime() as t:
            score = self.scoring(self.estimator_, self.X_ref_, self.y_ref_)
        scores = {
            "eval_time": t(),
            "step": len(self.results_),
            "CB_size": len(self.estimator_)}
        if isinstance(score, dict) and self.refit:
            self.results_.append({**score, **scores})
            return self.results_[-1][self.refit]
        else:
            self.results_.append({"score": score, **scores})
            return self.results_[-1]["score"]

    def stopping_criterion(self, iteration) -> Tuple[bool, str]:
        """If this method returns True, the maintenance loop is interrupted.
        
        By default, early stopping (with patience) is the only considered condition"""

            # if enough iterations have been performed, check if the performance has improved in the last `patience` 
            # iterations` 
        patience_test = self.patience >= 0 and len(self.scores_) >= self.patience and max(self.scores_[-self.patience-1:-1]) > self.scores_[-1]
        if patience_test: stopping_reason = "early stopping"
        else: stopping_reason = ""
        return patience_test, stopping_reason

    def fit(self,
            X: Iterable[SourceSpaceElement],
            y: Iterable[OutcomeSpaceElement],
            X_ref: Iterable[OutcomeSpaceElement]=None,
            y_ref: Iterable[OutcomeSpaceElement]=None,
            classes: List[OutcomeSpaceElement]="infer",
            n_iter: Optional[int]=None,
            warm_start: bool=False,
            force_copy: bool=True,
            fit_kwargs=dict(),
            **kwargs) -> Self:
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
        
        # things to do before the loop
        self.before_fit_loop(
            X,
            y,
            X_ref=X_ref,
            y_ref=y_ref,
            classes=classes,
            n_iter=n_iter,
            warm_start=warm_start,
            force_copy=force_copy,
            fit_kwargs=fit_kwargs,
            **kwargs)

        # start the fitting process
        if self.memorize_estimators:
            self.estimators_ = [clone(self.estimator_)]
            self.best_estimator_ = self.estimators_[0]
        else:
            self.best_estimator_ = clone(self.estimator_)

        # prepare the scoring process
        self.results_ = []
            
        check_is_fitted(self.estimator_)
        self.scores_ = [self.get_score_to_watch()]
        self.best_score_ = self.scores_[0]
        self.best_index_ = 0

        # main loop
        self.stopping_reason_ = "Reached n_iter"
        for i in range(self.n_iter_):
            # update the estimator
            try:
                with catchtime() as t:
                    self.update()
            except ValueError as e:
                if "All-NaN slice encountered" in str(e):
                    self.stopping_reason_ = "All-NaN slice encountered in update"
                    break
                else: 
                    self.stopping_reason_ = e
                    raise e
            
            # evaluate the latest model and update best model
            self.scores_.append(self.get_score_to_watch())
            self.results_[-1]["fit_time"] = t()

            if self.memorize_estimators:
                # un-CUDA the estimator to be stored
                if "device_" in self.estimator_.__dict__.keys():
                    estimator_for_memorize = clone(self.estimator_)
                    estimator_for_memorize.to_device("cpu")
                else:
                    estimator_for_memorize = clone(self.estimator_)
                self.estimators_.append(estimator_for_memorize)
            if self.scores_[-1] > self.best_score_: # update best model
                self.best_score_ = self.scores_[-1]
                if self.memorize_estimators:
                    self.best_estimator_ = self.estimators_[-1]
                else:
                    self.best_estimator_ = self.estimator_
                self.best_index_ = i+1  

            # check if the loop must be stopped
            stopping_criterion, stopping_reason = self.stopping_criterion(i)
            if stopping_criterion:
                self.stopping_reason_ = stopping_reason
                break
        

        self.estimator_ = self.best_estimator_

        self.after_fit_loop()

        check_is_fitted(self.estimator_)
        return self#.estimator#, self.best_score_

    def score(self, X, y, sample_weight: None=None) -> float:
        check_is_fitted(self.estimator_)
        return self.estimator_.score(X, y, sample_weight=sample_weight)

    def update(self) -> None:
        raise NotImplementedError()