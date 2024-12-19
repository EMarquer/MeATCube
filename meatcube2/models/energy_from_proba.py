"""Uses any sklearn model equipped with predict_proba to obtain the energy"""
from __future__ import annotations # for self-referring type hints
import torch, numpy as np
import pandas as pd
from typing import Any, Union, Literal, Tuple, Optional, Callable, Generic, TypeVar, Iterable, List
from typing_extensions import Self
from collections.abc import Sequence
from scipy.spatial.distance import squareform, pdist, cdist
from tqdm.auto import tqdm
import pickle
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from logging import warning

from meatcube2.models.AbstractEnergyBasedPredictor import ACaseBaseEnergyPredictor

from .energy_knn_backend import KNNEnergyComputations
from ..utils import to_numpy_array, pairwise_dist, cart_dist
from ..torch_utils import remove_index, append_symmetric
from .AbstractEnergyBasedClassifier import ACaseBaseEnergyClassifier, SourceSpaceElement, OutcomeSpaceElement
import inspect

NumberOrBool = Union[float, int, bool]

class EnergyClf(ACaseBaseEnergyClassifier):
    """
    
    Attributes
    ----------
    model
    """
    model: ClassifierMixin = None
    fit_kwargs_ = None
    
    def __init__(self, model):
        """
        Parameters
        ----------
        """
        self.model = model
        assert callable(getattr(model, "predict_proba", None)), "model should implement predict_proba, but the method was not found"
        self.predict_proba = model.predict_proba
        if callable(getattr(model, "predict", None)): self.predict = model.predict
        
        
    def fit(self, 
            X: Iterable[SourceSpaceElement],
            y: Iterable[OutcomeSpaceElement],
            classes: List[OutcomeSpaceElement] | Literal['infer'] = "infer",
            force_copy=True,
            **model_fit_kwargs) -> Self:
        super().fit(X, y, classes, force_copy)

        # filter out unknown kwargs
        args, varargs, varkw, defaults = inspect.getargspec(self.model.fit)
        if not varkw: 
            model_fit_kwargs = {k:v for k,v in model_fit_kwargs.items() if k in args}
        self.fit_kwargs_ = model_fit_kwargs
        self.model.fit(X, y, **model_fit_kwargs)
        return self

    def remove(self, index: int) -> EnergyClf:
        """Returns a copy of this MeATCubeCB object where case `index` has been removed.
        
        If initialized, will copy and update the similarity matrices and the cube."""
        check_is_fitted(self)
        
        updated_model = EnergyClf(clone(self.model))
        updated_model.fit(
            X=np.delete(self._X, index, axis=0),
            y=np.delete(self._y, index, axis=0),
            classes=self.classes_,
            device=self.fit_kwargs_)

        check_is_fitted(updated_model)
        return updated_model
    
    def add(self, case_source: SourceSpaceElement, case_outcome: OutcomeSpaceElement) -> EnergyClf:
        """Returns a copy of this MeATCubeCB object where case `index` has been removed.
        
        If initialized, will copy and update the similarity matrices and the cube."""
        check_is_fitted(self)
        updated_model = EnergyClf(clone(self.model))
        if case_outcome not in self.classes_:
            classes = self.classes_ + [case_outcome]
        else:
            classes = self.classes_
        updated_model.fit(
            X=np.append(self._X, [case_source], axis=0),
            y=np.append(self._y, [case_outcome], axis=0),
            classes=classes,
            device=self.fit_kwargs_)
        
        check_is_fitted(updated_model)
        return updated_model

###########################################################
# TODO: check output shape

    def energy_cb(self, as_tensor=False):
        energies: np.ndarray = self.model.predict_proba(self._X)
        indices: np.ndarray = self._outcome_index(self._y)
        energy = np.take(energies, indices, axis=1).mean()

        return energy
    
    def energy_case_from_cb(self, index: int, as_tensor=False):
        while index < 0:
            index += len(self)
        energies: np.ndarray = self.model.predict_proba(self._X[index:index+1])
        indices: np.ndarray = self._outcome_index(self._y[index:index+1])
        energy = np.take(energies, indices, axis=1).mean()
        
        return energy
    
    def energy_case_new(self, X: SourceSpaceElement, y: OutcomeSpaceElement, as_tensor=False) -> float:
        if y not in self.classes_:
            return self.add(X, y).energy_case_from_cb(-1, as_tensor=as_tensor)
        energies: np.ndarray = self.model.predict_proba(np.array([X]))
        indices: np.ndarray = self._outcome_index(np.array([y]))
        energy = np.take(energies, indices, axis=1).mean()

        return energy
    def energy_cases_new(self,
                         X: Iterable[SourceSpaceElement],
                         y: Iterable[OutcomeSpaceElement], as_tensor=False) -> torch.FloatTensor:
        energies: np.ndarray = self.model.predict_proba(X)
        indices: np.ndarray = self._outcome_index(y)

        # identify indices that do not belong to the known classes, and replace with class 0
        unk_indices = indices == None
        indices = np.select([~unk_indices], [indices], default=0).astype(int)

        energy = energies.take(indices, axis=1)
        
        # replace energies corresponding to unknown classes with +inf
        energy[unk_indices.reshape(1, -1).repeat(energies.shape[0], axis=0)] = np.inf
        

        return energy

###########################################################
    

    def _outcome_index(self, outcome: Union[OutcomeSpaceElement, Iterable[OutcomeSpaceElement]]) -> np.ndarray:
        if isinstance(self.model.classes_, np.ndarray):
            index = lambda x: np.where(self.model.classes_==x)[0][0] if np.any(self.model.classes_==x) else None
        else:
            index = lambda x: self.model.classes_.index(x) if x in self.model.classes_ else None
        return np.array([index(o) for o in outcome])
        


    #######################################################
    def _more_tags(self):
        return {
            "_xfail_checks": {
                # check_estimator checks that fail:
                "check_complex_data": "AssertionError: Did not raise: [<class 'TypeError'>]",
            },
            "allow_nan": False,
            "poor_score": True,
            "requires_y": True,
            "X_types": ["2darray", "sparse", "categorical", "1dlabels", "2dlabels", "string"]
        }