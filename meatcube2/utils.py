
import torch
import numpy as np
import pandas as pd
from typing import Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List
from scipy.spatial.distance import squareform, pdist, cdist
from tqdm.auto import tqdm

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import available_if
from sklearn.base import MetaEstimatorMixin
from sklearn.model_selection._search import _estimator_has

def to_numpy_array(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.numpy()
    elif isinstance(values, (pd.DataFrame, pd.Series)):
        return values.to_numpy()
    elif isinstance(values, np.ndarray):
        return values
    else:
        try: # float array-like
            return np.array(values, dtype=float)
        except ValueError: # non-float array-like
            return np.array(values)
        



# def torch_cdist():
#     pass

def pairwise_dist(
        data: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
        metric,
        **metric_kwargs):
    """A wrapper for scipy.spatial.distance.pdist, which handles a torch-based version and an object version."""
    if isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
        if (data.dtype != object):
            return pdist(data.reshape(-1,data[0].size), metric=metric)
        else:
            n = data.shape[0]
            out_size = (n * (n - 1)) // 2
            dm = np.ndarray(dtype=np.double, shape=(out_size,))
            k = 0
            for i in range(data.shape[0] - 1):
                for j in range(i + 1, data.shape[0]):
                    dm[k] = metric(data[i], data[j], **metric_kwargs)
                    k += 1
            return dm
    elif isinstance(data, torch.Tensor):
        raise NotImplementedError
    else:
        raise ValueError
def cart_dist(a: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
              b: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
              metric, **metric_kwargs):
    """A wrapper for scipy.spatial.distance.cdist, which handles a torch-based version and an object version."""
    if isinstance(a, (np.ndarray, pd.DataFrame, pd.Series)) and isinstance(b, (np.ndarray, pd.DataFrame, pd.Series)):
        if (a.dtype != object) and (b.dtype != object):
            return cdist(
                a.reshape(-1, a[0].size),
                b.reshape(-1, a[0].size),
                metric=metric)
        else:
            n = a.shape[0]
            m = b.shape[0]
            dm = np.ndarray(dtype=np.double, shape=(n, m))
            for i in range(n - 1):
                for j in range(m - 1):
                    dm[i,j] = metric(a[i], b[j], **metric_kwargs)
            return dm
    elif isinstance(a, torch.Tensor):
        raise NotImplementedError
    else:
        raise ValueError

class MetaEstimatorScoreMixin(MetaEstimatorMixin):
    
    @available_if(_estimator_has("score_samples"))
    def score_samples(self, X):
        """Call score_samples on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``score_samples``.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements
            of the underlying estimator.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            The ``best_estimator_.score_samples`` method.
        """
        check_is_fitted(self)
        return self.best_estimator_.score_samples(X)

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted labels or values for `X` based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class probabilities for `X` based on the estimator with
            the best found parameters. The order of the classes corresponds
            to that in the fitted attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.best_estimator_.predict_proba(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class log-probabilities for `X` based on the estimator
            with the best found parameters. The order of the classes
            corresponds to that in the fitted attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.best_estimator_.predict_log_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_score : ndarray of shape (n_samples,) or (n_samples, n_classes) \
                or (n_samples, n_classes * (n_classes-1) / 2)
            Result of the decision function for `X` based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.decision_function(X)

    @available_if(_estimator_has("transform"))
    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
            `X` transformed in the new space based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.transform(X)

    @available_if(_estimator_has("__len__"))
    def __len__(self):
        check_is_fitted(self)
        return self.best_estimator_.__len__()
    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, Xt):
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        ----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Result of the `inverse_transform` function for `Xt` based on the
            estimator with the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.inverse_transform(Xt)

    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`.

        Only available when `refit=True`.
        """
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the search estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        return self.best_estimator_.n_features_in_

    @property
    def classes_(self):
        """Class labels.

        Only available when `refit=True` and the estimator is a classifier.
        """
        _estimator_has("classes_")(self)
        return self.best_estimator_.classes_