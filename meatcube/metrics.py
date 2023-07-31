from typing import Literal
import torch
from sklearn.metrics import (
    precision_recall_fscore_support as precision_recall_fscore_support_,
    f1_score as f1_score_
)

try:
    from .meatcubecb import MeATCubeCB
except ImportError as e:
    try:
        from meatcubecb import MeATCubeCB
    except ImportError:
        raise e
    
def confidence(cb: MeATCubeCB, X, keepdim=True):
    """Returns a tensor with, for each possible outcome, the confidence (as defined for energy-based models) of the CB in the prediction for `sources`.
    
    If `keepdim=True`, the confidence is a [|R|, |S|] matrix,
    with for each outcome `r∈R` and each source `s∈S`, the confidence if `r` is predicted for `s` and 0 otherwise.
    
    If `keepdim=False`, the confidence is a [|S|] vector, with for each source `s∈S`, the confidence in the predicted outcome.
    """
    outcomes, logits = cb.predict(X, return_logits=True, return_outcome_indices=True)

    conf = torch.zeros_like(logits).transpose(-1, -2)
    for outcome in cb.potential_outcomes:
        outcome_id = cb.outcome_index(outcome)
        mask = outcomes == outcome_id
        sorted_logits = logits[mask].sort(dim=-1, descending=False).values.transpose(-1, -2)
        #conf[outcome, mask] = logits[mask,outcome] - sorted_logits[1]
        conf[outcome_id, mask] = sorted_logits[1] - sorted_logits[0]

    if keepdim:
        return conf
    else:
        return conf.sum(dim=0)

def accuracy(cb: MeATCubeCB, X, y):
    gold_labels = cb.outcome_index(y)
    pred_labels = cb.predict(X, return_outcome_indices=True)
    successes = pred_labels == gold_labels

    acc = successes.mean(dtype=float)
    return acc

def precision_recall_fscore_support(cb: MeATCubeCB, X, y, average: Literal['binary', 'micro', 'macro', 'samples', 'weighted']='macro'):
    gold_labels = cb.outcome_index(y)
    pred_labels = cb.predict(X, return_outcome_indices=True)
    return precision_recall_fscore_support_(gold_labels, pred_labels, average=average, zero_division=0)

def f1_score(cb: MeATCubeCB, X, y, average: Literal['binary', 'micro', 'macro', 'samples', 'weighted']='macro'):
    gold_labels = cb.outcome_index(y)
    pred_labels = cb.predict(X, return_outcome_indices=True)
    return f1_score_(gold_labels, pred_labels, average=average, zero_division=0)

def mce(cb: MeATCubeCB, X, y):
    return cb.competence(X, y, strategy='MCE', aggregation=None).to(float).mean()
def hinge(cb: MeATCubeCB, X, y, margin=0.01):
    return cb.competence(X, y, strategy='hinge', aggregation=None, margin=margin).to(float).mean()



def prediction_stats(cb: MeATCubeCB, X, y, margin=0.01, average: Literal['binary', 'micro', 'macro', 'samples', 'weighted']='macro'):
    """
    Returns: In order:
    - predicted labels
    - successes
    - accuracy
    - precision (micro average)
    - recall (micro average)
    - F1 (micro average)
    - F1 (macro average)
    - global MCE competence
    - global Hinge competence
    - per input MCE competence
    - per input Hinge competence
    """
    gold_labels = cb.outcome_index(y)
    pred_labels = cb.predict(X)
    successes = pred_labels == gold_labels
    acc = successes.mean(dtype=float)
    precision, recall, f1_micro, support = precision_recall_fscore_support_(gold_labels, pred_labels, average="micro", zero_division=0)
    f1_macro = f1_score_(gold_labels, pred_labels, average="macro", zero_division=0)

    mce = cb.competence(X, y, strategy='MCE', aggregation=None)
    hinge = cb.competence(X, y, strategy='hinge', aggregation=None, margin=margin)
    return successes, acc, precision, recall, f1_micro, f1_macro, mce.to(float).mean(), hinge.to(float).mean(), mce, hinge