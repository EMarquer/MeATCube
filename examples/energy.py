from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
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

# create the CB
source_similarity = lambda x,y: np.exp(- np.linalg.norm(x - y))
outcome_similarity = lambda x,y: (True if x == y else False)
cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)

# energy of the CB
energy = cb.energy()
print(f"Energy of the CB: {energy}")

# contribution to the energy of a particular case: case at index 1 (i.e. 2nd case) from the CB
energy_1 = cb.energy(index=1)
print(f"Energy contribution of case at index 1 from the CB: {energy_1}")


# contribution to the energy of a particular case: case at index 1 (i.e. 2nd case) from the CB, as if it was not in the CB
# - step 1: remove the case from the CB
cb_minus_1 = cb.remove(1)
# - step 2: computing the energy of the "new" case
energy_1_ = cb_minus_1.energy_new_case(X_train.iloc[1], y_train.iloc[1])
print(f"Energy contribution of case at index 1 when added to the CB: {energy_1_} (should be equal to {energy_1})")
