"""Following https://cora.ucc.ie/server/api/core/bitstreams/39193798-3fe0-461a-b1b6-3d9cffd108d3/content
The default similarity is as follows:
weighted average, for each attribute, of either:
- `matching` distance, for symbolic attributes
- `normalized error`, normalized by the min/max values of the attribute, for numeric attributes

The weights are decided ???
"""
from typing import Any, List
import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis

class MultivaluedSimilarity():
    def __init__(self, numeric_attributes: List[int], symbolic_attributes: List[int]) -> None:
        self.numeric_attributes = numeric_attributes
        self.symbolic_attributes = symbolic_attributes
        self.attributes = symbolic_attributes + numeric_attributes
        self.att_weights = [1. for att in self.attributes]
        self.att_sims = {
            **{att: NumericSimilarity() for att in numeric_attributes},
            **{att: SymbolicSimilarity() for att in symbolic_attributes}
        }

    def extract_mins_maxs(self, data: np.ndarray) -> None:
        for att in self.numeric_attributes:
            if att in data.shape[0]:
                self.att_sims[att].extract_min_max(data[att])

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Similarity between x and y, each containing a single record."""
        return (
            np.sum(weight * self.att_sims[att](x[att], y[att]) for att, weight in zip(self.attributes, self.att_weights)) /
            np.sum(self.att_weights)
        )
    
    def compute_without_aggregation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Similarity between x and y, each containing a single record."""
        return [self.att_sims[att](x[att], y[att]) for att in self.attributes]
    
    def fit(self, X: np.ndarray=None, y: np.ndarray=None, max_iter=100, random_state=42):
        """
        Finds ideal weights with NeighborhoodComponentsAnalysis

        :param X: data
        :param y: labels
        """
        nca = NeighborhoodComponentsAnalysis(n_components=1, random_state=random_state, max_iter=max_iter)
        nca.fit(X, y)

        #diag_values, diag_matrix = np.linalg.eig(nca.components_)
        #self.sim.att_weights = diag_values.tolist()
        self.att_weights = nca.components_[0].tolist()
        return nca.components_

class NumericSimilarity():
    def __init__(self) -> None:
        self.min = None
        self.max = None

    def extract_min_max(self, data: np.ndarray) -> None:
        """Data contains a sequence of record."""
        self.min = data.min()
        self.max = data.max()

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Similarity between x and y, each containing a single record."""
        if (self.min is not None) and (self.max is not None):
            return 1 - (np.abs(x - y) / (self.max - self.min))
        else:
            return 1 - np.abs(x - y)

class SymbolicSimilarity():
    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.equal(x, y).astype(float)