from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

USE_STRING_VALUES = True
RUN_EXAMPLE = 0 # 0 for all examples, or example number

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
from meatcube.maintenance import fancy_distillation_process

# create the CB
source_similarity = lambda x,y: np.exp(- np.linalg.norm(x - y))
outcome_similarity = lambda x,y: (True if x == y else False)
cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)

# performance of the CB
f1, acc = f1_score(cb, X_test, y_test), accuracy(cb, X_test, y_test)
print(f"Initial performance of the CB: F1 {f1}, Accuracy {acc}")

# --- Example 1 ---
# repeat removals (1 by 1) until nothing remains in the CB
print(f"Example: Repeat removals until nothing remains in the CB")
cb_ = fancy_distillation_process(cb, X_test, y_test,
                strategy="hinge",
                monitor="hinge",
                register=["F1", "accuracy"],
                margin=0,
                step_size=1,
                patience=len(cb))
f1, acc = f1_score(cb_, X_test, y_test), accuracy(cb_, X_test, y_test)
print(f"Final performance of the CB: F1 {f1}, Accuracy {acc} with a CB of size {len(cb_)}")

