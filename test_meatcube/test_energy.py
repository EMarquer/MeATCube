
import pytest
import numpy
import matplotlib.pyplot as plt

# load meatcube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import meatcube as mc



case_indices = list(range(50))
@pytest.mark.parametrize('case_index', case_indices)
def test_energy_1_case(iris_num_cb, iris_num_split, case_index):
    # energy of the case
    energy_1 = iris_num_cb.energy(index=case_index)
    X_train = iris_num_split[0]
    y_train = iris_num_split[2]

    # contribution to the energy of a particular case: case at index 1 (i.e. 2nd case) from the CB, as if it was not in the CB
    # - step 1: remove the case from the CB
    cb_minus_1 = iris_num_cb.remove(case_index)
    # - step 2: computing the energy of the "new" case
    energy_1_ = cb_minus_1.energy_new_case(X_train.iloc[case_index], y_train.iloc[case_index])

    assert energy_1_.item() == energy_1