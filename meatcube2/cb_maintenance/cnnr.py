
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence, Iterable
from functools import partial, reduce
from itertools import product
import numbers
import operator
import time
import warnings
from logging import warning, error

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
from sklearn.neighbors import KNeighborsClassifier

from copy import deepcopy as clone

from ..models.AbstractEnergyBasedClassifier import ACaseBaseEnergyClassifier, OutcomeSpaceElement, SourceSpaceElement
from ..metrics import clf_prediction_summary
from ..utils import catchtime
from ..utils import MetaEstimatorScoreMixin as MetaEstimatorMixin
from .energy_compress import CBClassificationMaintainerBase

class CNNR(CBClassificationMaintainerBase):
    """
    Condensed Nearest Neighbor Rule, original 1NN version, as of Hart, 1968
    https://sci2s.ugr.es/keel/pdf/algorithm/articulo/hart1968.pdf

    There are 
    self.changes_: bool, if False interrupt process
    self.candidate_cases_: in the decrement step, removal candidates are taken from this list ("grabbag" in Hart, 1968)
    self.kept_cases_: at the end of the process, all cases that have been kept for the final CB end up here ("store" in Hart, 1968)
    self.removed_cases_: at the end of the process, all cases that have been removed end up here

    
    To summarize the process:
    1. before_fit_loop:
        initialize all state variable (changes_, candidate_cases_, kept_cases_, removed_cases_, ...)
        take first case to initialize CB
    2. while not stopping_criterion and i in self.candidate_cases_:
        Strictly following (Hart, 1968), we have
        1. predict sample i using self.kept_cases_
        2. correctly predicted, add to self.candidate_cases_
        3. if not correctly predicted, add to self.kept_cases_

        Another (equivalent) description is: 
        1. predict a sample i using self.kept_cases_
        2. if correctly predicted, do nothing
        3. if not correctly predicted, add to self.kept_cases_ and remove from self.candidate_cases_
    3. after_fit_loop:
        cleanup all state variable (changes_, ...)

    stopping criterion is described in (Hart, 1968), point 4. as any of:
    - grabbag exhausted (in our case end of iteration)
    - no change (in our case "not self.changes_")
    """

    def __init__(self,
                 estimator: ACaseBaseEnergyClassifier,
                 memorize_estimators: bool=False,
                 scoring: Union[str, callable]=clf_prediction_summary,
                 refit: str="accuracy",
                 patience: int=-1,
                 random_state=42,
                 n_neighbors=1,
                 use_knn_proxy=True,
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
            If `mode`=="decrement" or `mode`=="increment", `patience` is the maximum number of iterations of the 
            decremental (or incremental) process that we wait once no improvement on the score is observed.
            If `patience` steps go on (within the limits of `n_iter`) without improvements on the score, the process is 
            interrupted and the best ... is used.
        use_knn_proxy : bool (default=True)
            If True, works as defined in CNNR, using a kNN (k=n_neighbors).
            If False, uses the prediction of the estimator to determine the cases to add
            """
        super().__init__(
            estimator=estimator,
            memorize_estimators=memorize_estimators,
            scoring=scoring,
            refit=refit,
            patience=patience,
            random_state=random_state,
        )
        self.n_neighbors = n_neighbors
        self.use_knn_proxy = use_knn_proxy

    def before_fit_loop(self, X, y, X_ref, y_ref, classes = "infer", mode = "init", increment_init = None, n_iter = None, warm_start = False, force_copy = True, fit_kwargs=dict(), **loss_kwargs):
        self.changes_ = True
        self.candidate_cases_ = list(range(len(X)))
        self.removed_cases_ = []

        super().before_fit_loop(X, y, X_ref, y_ref, classes=classes, n_iter=n_iter, warm_start=warm_start, force_copy=force_copy, fit_kwargs=fit_kwargs)

            
        # find the maximum number of iterations
        if self.n_iter_ is None:
            self.n_iter_ = len(self.X_) - 1
        else:
            self.n_iter_ = min(self.n_iter_, len(self.X_) - 1)

        # initialize the CB with the first case
        self.kept_cases_ = self.candidate_cases_[:1]
        self.candidate_cases_ = self.candidate_cases_[1:]
        self.X_kept_, self.y_kept_ = self.X_[:1], self.y_[:1]

        # initialize the kNN with the first case
        if self.use_knn_proxy:
            self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            self.knn_.fit(self.X_kept_, self.y_kept_)
        else:
            self.knn_ = None

        # try to fit the estimator
        estimator_not_fitted = True
        pre_iter = 0
        while estimator_not_fitted and pre_iter < self.n_iter_:
            try:
                self.estimator_.fit(self.X_kept_, self.y_kept_)
                estimator_not_fitted = False
            except Exception as e: # ignore fitting errors
                if not self.use_knn_proxy:
                    raise e
                self.update()
                pre_iter += 1
                if pre_iter == self.n_iter_:
                    raise e
        if pre_iter>0:
            warning(f"Had to perform {pre_iter} iteration{'s' if pre_iter > 1 else ''} before the estimator could be fitted (likely due to CB size too small or class under-representation), subtracting {'those iterations' if pre_iter > 1 else 'this iteration'} from the allowed iterations")
            self.n_iter_ -= pre_iter
        

    def after_fit_loop(self):
        del self.changes_
        super().after_fit_loop()

    def stopping_criterion(self, iteration) -> Tuple[bool, str]:
        stopping_criterion, stopping_reason = super().stopping_criterion(iteration) or not self.change_
        if not self.changes_:
            stopping_criterion = True
            stopping_reason = "No change this iteration" if stopping_reason!= "early stopping" else "No change this iteration (and early stopping)"
        return stopping_criterion, stopping_reason
        
    def update(self) -> None:
        """Strictly following (Hart, 1968), we have
        1. predict sample i using self.kept_cases_ using a kNN (default k = 1)
        2. correctly predicted, add to self.candidate_cases_
        3. if not correctly predicted, add to self.kept_cases_

        Another (equivalent) description is: 
        1. predict a sample i using self.kept_cases_ using a kNN (default k = 1)
        2. if correctly predicted, do nothing
        3. if not correctly predicted, add to self.kept_cases_ and remove from self.candidate_cases_
        """
        
        
        self.changes_ = False
        misclassified = []

        # step: for any cases that has not been processed yet, check if the current CB can predict it correctly
        # initial_index stands for the index in the initial, unprocessed state of the CB
        for i, initial_index in enumerate(self.candidate_cases_):
            (X_i, y_i) = (self.X_[i], self.y_[i])

            if self.use_knn_proxy:
                y_pred = self.knn_.predict(X_i[None, :])[0]
            else:
                y_pred = self.estimator_.predict(X_i[None, :])[0]

            # if it is misclassified, add the case to the CB, to self.kept_cases_, and remove from self.candidate_cases_
            if y_pred != y_i:
                self.kept_cases_.append(initial_index)
                misclassified.append(initial_index)
                # self.X_kept_ = np.vstack((self.X_kept_, X_i))
                # self.y_kept_ = np.vstack((self.y_kept_, y_i))
                self.X_kept_ = self.X_[self.kept_cases_]
                self.y_kept_ = self.y_[self.kept_cases_]
                if self.use_knn_proxy:
                    self.knn_.fit(self.X_kept_, self.y_kept_)
                else:
                    self.estimator_.fit(self.X_kept_, self.y_kept_)
                self.changes_ = True

        # step: remove cases that have been added from self.candidate_cases_
        self.candidate_cases_ = [i for i in self.candidate_cases_ if i not in misclassified]
        # update removed cases
        self.removed_cases_ = [i for i in range(self.X_.shape[0]) if i not in self.kept_cases_]

        # update the actual estimator
        if self.use_knn_proxy:
            try:
                self.estimator_.fit(self.X_kept_, self.y_kept_)
            except Exception as e:
                pass
        
    def after_fit_loop(self):
        del self.changes_
        return super().after_fit_loop()
        
