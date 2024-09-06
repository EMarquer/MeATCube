"""
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
"""
import pytest

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils.estimator_checks import check_estimator

# load meatcube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from meatcube2.meatcube_torch import MeATCubeCB

@pytest.mark.xfail
@pytest.mark.parametrize('case_index', [0])
def test_check_estimator(iris_num_cb: MeATCubeCB, iris_num_split, case_index):
        # energy of the case
        check_estimator(iris_num_cb)

case_indices = list(range(50))
@pytest.mark.parametrize('case_index', case_indices)
def test_energy_1_case(iris_num_cb: MeATCubeCB, iris_num_split, case_index):
        # energy of the case
        energy_1 = iris_num_cb.energy_case_from_cb(index=case_index)
        X_train, X_test, y_train, y_test = iris_num_split

        #check_estimator(iris_num_cb)
        #clf = MeATCubeCB()
        iris_num_cb.fit(X_train.to_numpy(), y_train.to_numpy())
        score = iris_num_cb.score(X_test.to_numpy(), y_test.to_numpy())
        # DecisionBoundaryDisplay.from_estimator(
        #         clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        #         )
        
        return (score, y_test)
