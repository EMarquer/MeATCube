from typing import Literal, Union, Tuple
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
    
    See also: https://bharathpbhat.github.io/2021/04/04/getting-confidence-estimates-from-neural-networks.html

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

#def loss_functional_values(cb: ACaseBaseEnergyClassifier, X, y, strategy: Literal["MCE", "hinge", "all"]="all", margin: float=0.1, **kwargs) -> Union[float, Tuple[float, float]]:
    # pred, energies = cb.predict_multiple(X, candidate_classes=cb.classes_, return_energies=True, **kwargs)

    # # check labels that are in the expected classes and find `rₜ' ≠ rₜ`: candidate_classes = {rₜ' ∈ self.classes_ | rₜ' ≠ rₜ}
    # contains_mask = np.vectorize(cb.classes_.__contains__)(y)

    # # energy for outcomes that are contained in cb.classes_:
    # # we need to select the energy of the gold label and the best energy among the rest
    # if np.count_nonzero(contains_mask) > 0:
    #     energies_contained = energies[contains_mask]
    #     gold_y_contained_ids = np.vectorize(cb.classes_.index)(y[contains_mask]) # index of rₜ, the correct labels in the energies, if the label is among the candidate labels
    #     next_best_y_contained_ids = np.argsort(energies_contained[:,~gold_y_contained_ids], axis=-1)[:,0] # sort in increasing, then take the index rₜ' of the smallest energy
    #     gold_energy_contained = energies_contained[gold_y_contained_ids]
    #     next_best_energy_not_contained = energies_contained[next_best_y_contained_ids]
    #     mce_contained = gold_energy_contained - next_best_energy_not_contained # E(CB ∪ {(sₜ, rₜ)} - (min_{rₜ' ≠ rₜ}E(CB ∪ {(sₜ, rₜ')}))
            
    # # energy for outcomes that are not contained in cb.classes_
    # # we need to compute the energy of the gold label separately, and get the best energy among cb.classes_
    # if np.count_nonzero(~contains_mask) > 0:
    #     gold_energy_not_contained = np.vectorize(lambda x_, y_: cb.energy_case_new(x_, y_, **kwargs))(X[~contains_mask], y[~contains_mask])
    #     next_best_energy_not_contained = np.sort(energies_contained[~contains_mask], axis=-1)[:,0] # sort in increasing, then take the index rₜ' of the smallest energy
    #     mce_not_contained = gold_energy_not_contained - next_best_energy_not_contained

    # # gather all the MCE values
    # if np.count_nonzero(~contains_mask) > 0 and np.count_nonzero(~contains_mask) > 0:
    #     mce = np.concatenate([mce_contained, mce_not_contained])
    # elif np.count_nonzero(~contains_mask) > 0:
    #     mce = mce_not_contained
    # else:
    #     mce=mce_contained
    # mce_score: float = mce.mean()

    # if strategy in {"hinge", "all"}:
    #     # use the element-wise MCE to obtain the hinge
    #     hinge = np.maximum(0, margin + mce)
    #     hinge_score: float = hinge.mean()

    # if   strategy == "MCE": return pred, mce_score
    # elif strategy == "hinge": return pred, hinge_score
    # elif strategy == "all": return pred, mce_score, hinge_score


def clf_prediction_summary(cb: ACaseBaseEnergyClassifier, X, y, margin=0.01):
    """
    Returns: 
    
    """
    scores = dict()

    # traditional scores, that are computed without needing access to the energy
    gold_labels = y
    pred_probas = cb.predict_proba(X)
    pred_labels = cb.predict(X)
    #pred_labels = np.vectorize(cb.classes_.__getitem__)(np.argmax(pred_probas))
    scores = {"gold_labels": gold_labels, "pred_probas": pred_probas, "pred_labels": pred_labels}
    accuracy = accuracy_score(gold_labels, pred_labels)
    scores["accuracy"] = accuracy
    precision, recall, f1, support = precision_recall_fscore_support(gold_labels, pred_labels, average=None, zero_division=0)
    scores["per_class_f1"] = f1
    scores["per_class_precision"] = precision
    scores["per_class_recall"] = recall
    scores["per_class_support"] = support
    for average in {"macro", "micro", "weighted"}:
        precision, recall, f1, _ = precision_recall_fscore_support(gold_labels, pred_labels, average=average, zero_division=0)
        scores[average+"_f1"] = f1
        scores[average+"_precision"] = precision
        scores[average+"_recall"] = recall

    # scores computed with the energy
    mce = cb.loss_cb(X, y, strategy="MCE", aggregation=None)
    scores["MCE_loss_functionals"] = mce
    scores["MCE_loss"] = np.mean(scores["MCE_loss_functionals"])
    scores["hinge_loss_loss_functionals"] = np.maximum(0, margin + mce)
    scores["hinge_loss"] = np.mean(scores["hinge_loss_loss_functionals"])
    
    return scores