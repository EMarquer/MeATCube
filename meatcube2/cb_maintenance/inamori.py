
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
from .cb_maintainer import CBClassificationMaintainerBase


from .inamoriISel import base, cnn, enn, icf, lssm, lsbo, drop3, ldis, cdis, xldis, psdsp, ib3, cis, egdis
INAMORY_I_SEL_METHODS = ['cnn', 'enn', 'icf', 'lssm', 'ldis', 'cdis', 'xldis', 'psdsp', 'ib3', 'egdis', 'cis']
INAMORY_I_SEL_OPTIONS = Literal['cnn', 'enn', 'icf', 'lssm', 'ldis', 'cdis', 'xldis', 'psdsp', 'ib3', 'egdis', 'cis']
def get_selector(method: str) -> base.InstanceSelectionMixin:

    if method == 'cnn':     return cnn.CNN()
    if method == 'enn':     return enn.ENN()
    if method == 'icf':     return icf.ICF()
    if method == 'lssm':    return lssm.LSSm()
    if method == 'ldis':    return ldis.LDIS()
    if method == 'cdis':    return cdis.CDIS()
    if method == 'xldis':   return xldis.XLDIS()
    if method == 'psdsp':   return psdsp.PSDSP()
    if method == 'ib3':     return ib3.IB3()
    if method == 'egdis':   return egdis.EGDIS()
    if method == 'cis':     return cis.CIS(task="atc")

    raise ValueError("Unkown instance selector for inamori1932/instance-selection-approaches: {method}")

class InamoriISelSingleStep(CBClassificationMaintainerBase):
    """
    Applies the algorithms from inamori1932/instance-selection-approaches iSel folder as single step processes.
    """
    method: INAMORY_I_SEL_OPTIONS

    def __init__(self,
                 estimator: ACaseBaseEnergyClassifier,
                 memorize_estimators: bool=False,
                 scoring: Union[str, callable]=clf_prediction_summary,
                 refit: str="accuracy",
                 patience: int=None,
                 random_state=42,
                 method: INAMORY_I_SEL_OPTIONS='cnn',
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
            """
        super().__init__(
            estimator=estimator,
            memorize_estimators=memorize_estimators,
            scoring=scoring,
            refit=refit,
            patience=-1,
            random_state=random_state,
        )
        self.method=method

    def before_fit_loop(self, X, y, X_ref, y_ref, classes = "infer", mode = "init", increment_init = None, n_iter = None, warm_start = False, force_copy = True, fit_kwargs=dict(), **loss_kwargs):
        super().before_fit_loop(X, y, X_ref, y_ref, classes=classes, n_iter=1, warm_start=warm_start, force_copy=force_copy, fit_kwargs=fit_kwargs)

        # initialize the CB
        self.kept_cases_ = list(range(len(X)))
        self.removed_cases_ = []
        self.X_kept_, self.y_kept_ = self.X_, self.y_

        # try to fit the estimator
        estimator_not_fitted = True
        pre_iter = 0
        while estimator_not_fitted and pre_iter < self.n_iter_:
            try:
                self.estimator_.fit(self.X_kept_, self.y_kept_)
                estimator_not_fitted = False
            except Exception as e: # ignore fitting errors
                warning(f"Estimator could not be fitted on full dataset")
                raise e

    def stopping_criterion(self, iteration) -> Tuple[bool, str]:
        stopping_criterion, stopping_reason = False, ""
        if iteration >= 0: 
            stopping_criterion = True
            stopping_reason = f"End of instance selection algorithm ({self.method})"
        
        return stopping_criterion, stopping_reason
        
    def update(self) -> None:
        # from https://github.com/inamori1932/instance-selection-approaches/blob/main/example.ipynb
        selector = get_selector(method=self.method)
        selector.fit(self.X_, self.y_)
        idx = selector.sample_indices_
        self.kept_cases_ = idx
        self.removed_cases_ = set(range(len(self.X_))).difference(idx)
        self.X_kept_, self.y_kept_ =  self.X_[idx], self.y_[idx]
        self.estimator_.fit(self.X_kept_, self.y_kept_)
        
        
