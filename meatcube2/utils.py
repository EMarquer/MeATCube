
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

from time import perf_counter
from contextlib import contextmanager

@contextmanager
def catchtime() -> Callable[[], float]:
    """A context manager to measure the time in seconds"""
    t1 = t2 = perf_counter() 
    yield lambda: t2 - t1
    t2 = perf_counter() 

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
        
def estimate_mem_utilisation(tensor: torch.Tensor = None, size: Union[int, torch.Size, Iterable] = None, dtype: Union[type, torch.dtype] = None) -> int :
    """Estimates the size in bytes for a given tensor"""
    assert tensor is not None or (size is not None and dtype is not None)
    if dtype is not None:
        element_size = torch.tensor([1], dtype=dtype).element_size()
    else:
        element_size = tensor.element_size()

    if size is not None:
        if isinstance(size, int):
            nelement = size
        elif isinstance(size, torch.Size):
            nelement = 1
            for i in size: nelement*=i
        elif isinstance(size, Iterable):
            nelement = 1
            for i in size: nelement*=i
        else:
            raise ValueError(f"Unknown handling for size {size} of type {type(size)}, expected int, torch.Size, or iterable of ints")
    else:
        nelement = tensor.nelement()
    return element_size * nelement

def check_mem_available(device: torch.device, bytes: int) -> bool:
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        free, total = torch.cuda.mem_get_info(device)
        return bytes <= free
    else:
        import psutil
        free = psutil.virtual_memory()
        return bytes <= free

def estimate_batch_size(device: torch.device, batch_element_bytes: int, max_batch_size: int=2**20, fixed_overhead: int=0) -> int:
    """Given a memory estimate finds the largest batch size (a power of 2) that can fit in the memory of the device.
    
    The batch size is the number of repetitions of the batch element that can fit in the memory.

    Parameters
    ----------
    device : torch.device
        device to check
    batch_element_bytes : int
        size/estimated memory consumption of one element in the batch
    max_batch_size : int
        max value to try for batch size, ensures termination of the loop
    fixed_overhead : int
        constant value added to the memory consumption

    See also
    --------
    estimate_mem_utilisation :
        useful to estimate the number of bytes of one element in the batch

    Returns
    -------
    Tuple[int, bool]
        found batch size, or 0 if there is not enough space to put even one element of the batch on GPU
    """
    if not check_mem_available(device, (batch_element_bytes) + fixed_overhead):
        return 0
    batch_size = 1
    while check_mem_available(device, (batch_size * batch_element_bytes) + fixed_overhead) and batch_size < max_batch_size:
        batch_size *= 2
    return batch_size // 2 # undo the last multiplication, as it resulted in a not-enough-memory decision

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