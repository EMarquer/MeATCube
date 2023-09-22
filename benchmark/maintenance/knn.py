
try:
    from .sim import MultivaluedSimilarity
except ImportError as e:
    try:
        from sim import MultivaluedSimilarity
    except ImportError:
        raise e
from collections import Counter
import numpy as np
import torch


class KNN():
    def __init__(self, sim: MultivaluedSimilarity, CB_source: np.ndarray=None, CB_outcome: np.ndarray=None) -> None:
        """
        
        Warning: modifies the self.sim object.

        :param CB_source: size [|CB|, |attributes|]
        :param CB_outcome: expected to contain the index of the class for each sample in the CB. size [|CB|]
        """
        self.sim = sim
        self.CB_source = CB_source
        self.CB_outcome = CB_outcome

    def classify(self, x: np.ndarray, k=-1):
        # 1. rank cases in CB by similarity
        if len(x.shape) >= 2:
            return [self.classify_one(x_, k) for x_ in x]
        else:
            return self.classify_one(x, k)
    
    def classify_one(self, x, k=-1):
        # 1. rank cases in CB by similarity
        counter = Counter()
        sims = np.array([self.sim(x, case) for case in self.CB_source])
        ranks = sims.argsort().argsort()
        for rank, sim, outcome in zip(ranks, sims, self.CB_outcome):
            if (
                (k > 0 and (rank >= ranks.max() - k)) or # consider only the top k most similar cases' outcome
                (k <= 0) # consider all cases' outcome
            ):
                counter[outcome] += sim # similarity = weight of vote
        return counter.most_common(1)[0][0] # return the value associated with the highest amount of vote
    
    def fit(self, CB_source: np.ndarray, CB_outcome: np.ndarray, max_iter=100, random_state=42):

        self.CB_source = CB_source
        self.CB_outcome = CB_outcome
        self.fit_weights_nca(max_iter=max_iter, random_state=random_state)

        return self

    def fit_weights_nca(self, CB_source: np.ndarray=None, CB_outcome: np.ndarray=None, max_iter=100, random_state=42):
        """
        Finds ideal weights with NeighborhoodComponentsAnalysis, then diagonalize the obtained linear transformation

        Warning: modifies the self.sim object.

        :param dev_source: size [|dev set|, |attributes|]
        :param dev_outcomes: expected to contain the index of the class for each sample in the dev set. size [|dev set|]
        """
        if CB_source is None or CB_outcome is None:
            CB_source = self.CB_source
            CB_outcome = self.CB_outcome

        return self.sim.fit(CB_source, CB_outcome, max_iter=max_iter, random_state=random_state)

    def fit_weights(self, dev_source: np.ndarray, dev_outcomes: np.ndarray, max_steps=2000, lr=1e-3, gamma=0.99):
        """
        Finds ideal weights with SGD.

        Use fit_weights_nca instead if possible

        Warning: modifies the self.sim object.

        :param dev_source: size [|dev set|, |attributes|]
        :param dev_outcomes: expected to contain the index of the class for each sample in the dev set. size [|dev set|]
        """

        # compute the similarities for each feature separately
        sims = np.ndarray((self.CB_source.shape[0], dev_source.shape[0], len(self.sim.attributes)))
        for i, CB_source_case in enumerate(self.CB_source):
            for j, dev_source_case in enumerate(dev_source):
                sims[i,j] = self.sim.compute_without_aggregation(CB_source_case, dev_source_case)
        # sims: [|CB|, |dev set|, num features]
        sims_ = torch.tensor(sims)
        # sims_: [|CB|, |dev set|, num features]

        # prepare the weights
        n_atts = self.CB_source.shape[1]
        weights_: torch.Tensor = torch.softmax(torch.ones(n_atts), dim=-1).view((1, 1, -1))
        weights_.requires_grad = True
        # weights_: [1, 1, num features]

        # compute the prediction
        # transform the class indices into one-hot encodings
        outcomes_one_hot = torch.arange(0, np.max(self.CB_outcome) + 1)
        CB_outcome_one_hot = (outcomes_one_hot.view(1, -1) == torch.tensor(self.CB_outcome).view(-1, 1))
        # CB_outcome_one_hot: [|CB|, |outcomes|]

        # === optimization ===
        optimizer = torch.optim.SGD([weights_], lr=lr, momentum=0.5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        best_loss = None
        best_weights = weights_/weights_.abs().sum()
        from tqdm import tqdm
        iter = tqdm(list(range(max_steps)), desc="Step")
        #for step in range(max_steps):
        for step in iter:
            optimizer.zero_grad()
            weights__ = weights_/(weights_.detach().abs().sum())
            sims_averaged = (sims_ * weights__).sum(dim=-1)
            # sims_averaged: [|CB|, |dev set|]

            votes = sims_averaged.unsqueeze(-1) * CB_outcome_one_hot.unsqueeze(-2)
            # votes: [|CB|, |dev set|, |outcomes|]

            logits = votes.sum(dim=0)
            # logits: [|dev set|, |outcomes|]

            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(dev_outcomes))
            loss.backward()

            if best_loss is None or loss < best_loss:
                best_weights = weights__.detach()
                best_loss = loss
            
            optimizer.step()
            scheduler.step()
            iter.set_postfix_str(f"Loss: {loss.item()}")

        weights_ = best_weights

        self.sim.att_weights = weights_.detach().view(-1).numpy().tolist()

def accuracy(knn: KNN, X_test, y_test, k=-1):
    if not isinstance(X_test, np.ndarray): X_test = X_test.to_numpy()
    if not isinstance(y_test, np.ndarray): y_test = y_test.to_numpy()
    pred = knn.classify(X_test, k=k)
    acc = (y_test == np.array(pred)).astype(float).mean()
    return acc

from sklearn.neighbors import KNeighborsClassifier
KNNSklearn = lambda dist, n_neighbors=5: KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric=dist)