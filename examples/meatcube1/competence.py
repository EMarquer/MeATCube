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
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import meatcube as mc

# create the CB
source_similarity = lambda x,y: np.exp(- np.linalg.norm(x - y))
outcome_similarity = lambda x,y: (True if x == y else False)
cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)

### Hinge Competence (default) ###
print("--- Hinge Competence ---")
# competence of the CB
comp = cb.competence(X_test, y_test) # same as cb.competence(X_test, y_test, strategy="hinge")
print(f"Competence of the CB: {comp}")

# contribution to the competence of a particular case: case at index 1 (i.e. 2nd case) from the CB
comp_1 = cb.competence(X_test, y_test, index=1)
print(f"Competence of case at index 1 from the CB: {comp_1}")

# contribution to the competence of particular cases: cases at index 1,2,3 (i.e. 2nd,3rd,4th cases) from the CB
comp_1_2_3 = cb.competence(X_test, y_test, index=[1, 2, 3])
print(f"Competence of cases at index 1,2,3 from the CB: {comp_1_2_3}")

### MCE Competence ###
print("--- MCE Competence ---")
# competence of the CB
comp = cb.competence(X_test, y_test, strategy="MCE")
print(f"Competence of the CB: {comp}")

# contribution to the competence of a particular case: case at index 1 (i.e. 2nd case) from the CB
comp_1 = cb.competence(X_test, y_test, index=1, strategy="MCE")
print(f"Competence of case at index 1 from the CB: {comp_1}")

# contribution to the competence of particular cases: cases at index 1,2,3 (i.e. 2nd,3rd,4th cases) from the CB
comp_1_2_3 = cb.competence(X_test, y_test, index=[1, 2, 3], strategy="MCE")
print(f"Competence of cases at index 1,2,3 from the CB: {comp_1_2_3}")