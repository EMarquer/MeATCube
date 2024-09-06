import pytest
import numpy
import matplotlib.pyplot as plt

# load meatcube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import meatcube as mc
from meatcube.plotting.cbscatter import plot_cb, plot_ref



"""To generate the baseline:
pytest -k "test_plot_cb" --mpl-generate-path test/baseline
"""
@pytest.mark.mpl_image_compare
def test_plot_cb(iris_num_cb):
    plot_cb(iris_num_cb)
    return plt.gcf()
@pytest.mark.mpl_image_compare
def test_plot_ref(iris_num_cb, iris_num_split):
    X_train, X_test, y_train, y_test = iris_num_split
    plot_ref(iris_num_cb, X_test, y_test)
    return plt.gcf()
@pytest.mark.mpl_image_compare
def test_plot_ref_pred(iris_num_cb, iris_num_split):
    X_train, X_test, y_train, y_test = iris_num_split
    plot_ref(iris_num_cb, X_test, y_test, iris_num_cb.predict(X_test))
    return plt.gcf()