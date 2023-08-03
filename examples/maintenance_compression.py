from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

USE_STRING_VALUES = True

iris = load_iris(as_frame=True)

X: pd.DataFrame = iris["data"] # source
y = iris["target"] # target

if USE_STRING_VALUES:
    # to test with strings as labels
    y = y.apply(lambda x: iris["target_names"][x]) 
    y_values = iris["target_names"]
else:
    y_values = np.unique(y)


# stratified splitting of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=42, stratify=y)

# add root directory to be able to import MeATCube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import meatcube as mc
from meatcube.metrics import f1_score, precision_recall_fscore_support, accuracy
from meatcube.maintenance import decrement, decrement_early_stopping

# create the CB
source_similarity = lambda x,y: np.exp(- np.linalg.norm(x - y))
outcome_similarity = lambda x,y: (True if x == y else False)
cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)

# performance of the CB
f1, acc = f1_score(cb, X_test, y_test), accuracy(cb, X_test, y_test)
print(f"Initial performance of the CB: F1 {f1}, Accuracy {acc}")



# remove 1 case from the initial CB
cb_minus_1, competences, result_index = decrement(cb, X_test, y_test,
                strategy="hinge",
                margin=0.1,
                k=1,
                return_all=True)
f1, acc = f1_score(cb_minus_1, X_test, y_test), accuracy(cb_minus_1, X_test, y_test)
print(f"Removed case {cb[result_index][0]} -> {cb[result_index][1]} which had a competence of {competences[result_index]}")
print(f"New performance of the CB: F1 {f1}, Accuracy {acc}")



# remove 10 cases from the initial CB
cb_minus_10, competences, result_index = decrement(cb, X_test, y_test,
                strategy="hinge",
                margin=0.1,
                k=10,
                return_all=True)
f1, acc = f1_score(cb_minus_10, X_test, y_test), accuracy(cb_minus_10, X_test, y_test)
print(f"Removed 10 cases:")
for result_idx in result_index:
    print(f"- {cb[result_idx][0]} -> {cb[result_idx][1]} which had a competence of {competences[result_idx]}")
print(f"New performance of the CB: F1 {f1}, Accuracy {acc}")
