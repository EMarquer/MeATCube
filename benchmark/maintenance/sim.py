"""Following https://cora.ucc.ie/server/api/core/bitstreams/39193798-3fe0-461a-b1b6-3d9cffd108d3/content
The default similarity is as follows:
weighted average, for each attribute, of either:
- `matching` distance, for symbolic attributes
- `normalized error`, normalized by the min/max values of the attribute, for numeric attributes

The weights are decided ???
"""
from typing import Any, List, Literal
import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier
from functools import cache
from scipy.special import softmax

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

    def set_uniform_weights(self):
        """Reset the weights to uniform"""
        self.att_weights = [1. for att in self.attributes]
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Similarity between x and y, each containing a single record."""
        return (
            np.sum(weight * self.att_sims[att](x[att], y[att]) for att, weight in zip(self.attributes, self.att_weights)) /
            np.sum(self.att_weights)
        )
    
    def compute_without_aggregation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Similarity between x and y, each containing a single record."""
        return [self.att_sims[att](x[att], y[att]) for att in self.attributes]
    
    def fit(self, X: np.ndarray=None, y: np.ndarray=None, max_iter=10000, random_state=42, method: Literal["nca", "gs", "none", None]=None, epsilon=1):
        """
        Finds ideal weights with NeighborhoodComponentsAnalysis

        :param X: data
        :param y: labels
        :param epsilon: when using method=None, epsilon is the minimal weight value before normalization
        """
        if method=="none" or method is None: # https://dergipark.org.tr/en/download/article-file/904927
            # Aᵢ(a) = {X[k] ∈ X: min(Cᵢ(a)) ≤ X[k][a] ≤ max(Cᵢ(a))}
            # Cᵢ(a): set of values for attribute a belonging to class i
            # Cᵢ(a) = (X[k][a] for k, xk in enumerate(X) if y[k]==Cᵢ)
            # for nominal data (defined by me): Aᵢ(a) = {X[k] ∈ X: X[k][a] ∈ Cᵢ(a)}
            #
            # Bᵢ(a) = Aᵢ(a) - ⋃"i≠j, j∈classes" Aⱼ(a)
            #
            # wₐ=|⋃"i∈classes" Bᵢ(a)| / n (in paper it was  wₐ=(⋃"i∈classes" |Bᵢ(a)|) / n, but it makes no sense)
            # n: len(X)
            #
            # wₐ*: normalized wₐ
            classes = np.unique(y)
            n = len(y)

            @cache
            def indices(i):
                """The indices of items in class a"""
                return {k for k, y_ in enumerate(y) if y_==i}

            @cache
            def C(i, a):
                return {X[k][a] for k in indices(i)}
                
                
            @cache
            def A(i, a):
                if a in self.numeric_attributes:
                    return {Xka for k in range(len(X)) if min(C(i, a)) <= (Xka:=X[k][a]) <= max(C(i, a))}
                else:
                    return {Xka for k in range(len(X)) if (Xka:=X[k][a]) in (C(i, a))}
                
            @cache
            def B(i, a):
                return A(i, a) - set().union(*[A(j, a) for j in classes if i!=j])
            
            def w(a):
                return len(set().union(*[B(i, a) for i in classes])) / n
            
            
            self.att_weights = []
            for a in self.attributes:
                self.att_weights.append(max(w(a), epsilon))
                #print([A(i, a) for i in classes])
                #  clear cache after each attribute
                C.cache_clear()
                A.cache_clear()
                B.cache_clear()

            #print(self.att_weights, n)

            # normalize
            total = sum(self.att_weights)
            self.att_weights = list(map(lambda x: x/total, self.att_weights))

            return self.att_weights

        elif method=="nca":
            nca = NeighborhoodComponentsAnalysis(n_components=1, random_state=random_state, max_iter=max_iter)
            nca.fit(X, y)

            #diag_values, diag_matrix = np.linalg.eig(nca.components_)
            #self.sim.att_weights = diag_values.tolist()
            self.att_weights = nca.components_[0].tolist()
            return nca.components_
        else: # Grid search with distance weights /!\ HAS BUGS /!\
            @cache
            def score_weights_knn(weights):
                dist = MultivaluedSimilarity(self.numeric_attributes, self.symbolic_attributes)
                dist.att_weights = weights
                knn = KNeighborsClassifier(n_neighbors=X.shape[0], metric = dist, weights='distance')
                return knn.score(X, y)
            
            # find combinations of weights
            iter_per_weight = np.floor(np.power((max_iter - 1), (1/len(self.att_weights)))).astype(int).item()
            print([list(range(1, iter_per_weight + 1))] * len(self.att_weights))
            np.stack(np.meshgrid(
                *([list(range(1, iter_per_weight + 1))] * len(self.att_weights)) # 
            ), -1)
            
            weight_matrix = np.stack([np.ones(len(self.att_weights))] + np.meshgrid(
                *([list(range(1, iter_per_weight + 1))] * len(self.att_weights)) # possible weights = 
            ), -1).T.reshape(-1,len(self.att_weights))


            softmax_weight_matrix = softmax(weight_matrix, axis=-1)
            no_duplicate_weight_matrix = np.unique(softmax_weight_matrix, axis=0)
            
            # find best set of weights
            scores = np.fromiter((score_weights_knn(weights_) for weights_ in no_duplicate_weight_matrix), float)

            # max score, corrsponding weight
            weights = no_duplicate_weight_matrix[np.argmax(scores)]
            self.att_weights = weights.tolist()

            return weights

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