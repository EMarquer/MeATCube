from typing import Literal
import torch
from scipy.special import softmax
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    accuracy_score
)

try:
    from .models import ACaseBaseEnergyClassifier
except ImportError as e:
    try:
        from models import ACaseBaseEnergyClassifier
    except ImportError:
        try:
            from meatcube2.models import ACaseBaseEnergyClassifier
        except ImportError:
            raise e
    
def confidence(cb: ACaseBaseEnergyClassifier, X, keepdim=True, batched=False, temperature=1):
    """Returns a tensor with, for each possible outcome, the confidence (i.e. the prediction probability) for `sources`.
    
    If `keepdim=True`, the confidence is a [|R|, |S|] matrix, with for each outcome `r∈R` and each source `s∈S`.
    
    If `keepdim=False`, the confidence is a [|S|] vector, with for each source `s∈S`, the confidence in the predicted outcome.
    """
    if batched:
        raise NotImplementedError("batched confidence computation not supported yet")
        # confidences = torch.stack([
        #     confidence(cb, X[], keepdim=True, batched=False) for 
        # ])
    else:
        outcome_probas = cb.predict_proba(X, temperature=temperature)
        
        if keepdim:
            return outcome_probas
        else:
            outcome_ids = np.argmax(outcome_probas, axis=-1, keepdims=True)
            return outcome_probas[outcome_ids]

def loss_functional_values(cb: ACaseBaseEnergyClassifier, X, y, strategy: Literal["MCE", "hinge", "all"]="all", **kwargs):

    y_pred, y_pred_ids, energies = cb.predict_multiple(X, return_id=True, return_energies=True, **kwargs)


    # find `rₜ' ≠ rₜ`: candidate_classes = {rₜ' ∈ self.classes_ | rₜ' ≠ rₜ}
    sorted_energies = energies.sort(axis=-1)
    r_t, r_t_prime = energies[y]


    # for (X_elem, y_elem) in zip(X, y):
    #     # find `rₜ' ≠ rₜ`: candidate_classes = {rₜ' ∈ self.classes_ | rₜ' ≠ rₜ}
    #     if cb.classes_.__contains__(y_elem):
    #         outcome_index = cb.classes_.index(y_elem)
    #         candidate_classes = cb.classes_[:outcome_index] + cb.classes_[outcome_index+1:]
    #     else:
    #         candidate_classes = cb.classes_

    #     mce = (
    #         cb.energy_case_new(X_elem, y_elem, **kwargs) # E(CB ∪ {(sₜ, rₜ)}
    #         - cb.predict_one(X_elem, candidate_classes=candidate_classes, return_energies=True, **kwargs)[1][0] # min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')})
    #     )
    #     hinge = max(0, margin + mce)
            
    #     pred_labels = cb.predict(X)
    # return

def prediction_summary(cb: ACaseBaseEnergyClassifier, X, y, margin=0.01, average: Literal['binary', 'micro', 'macro', 'samples', 'weighted']='macro'):
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
    gold_labels = y
    pred_labels = cb.predict(X)
    accuracy = accuracy_score(gold_labels)
    precision, recall, f1_micro, support = precision_recall_fscore_support(gold_labels, pred_labels, average="micro", zero_division=0)
    f1_macro = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)


    mce = cb.loss_functional_cb(X, y, strategy='MCE', aggregation=None)
    hinge = cb.loss_functional_cb(X, y, strategy='hinge', aggregation=None, margin=margin)
    return successes, acc, precision, recall, f1_micro, f1_macro, mce.to(float).mean(), hinge.to(float).mean(), mce, hinge