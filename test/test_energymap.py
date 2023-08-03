import pytest
import numpy
import matplotlib.pyplot as plt

# load meatcube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import meatcube as mc
from meatcube.plotting.preprocessing import prepare_ax
from meatcube.plotting.energymap import energymap
from meatcube.plotting.cbscatter import plot_cb, plot_ref



"""To generate the baseline:
pytest -k "test_energymap" --mpl-generate-path test/baseline
"""
@pytest.mark.mpl_image_compare
def test_energymap(iris_num_cb):
    energymap(iris_num_cb)
    return plt.gcf()

@pytest.mark.mpl_image_compare
def test_energymap_test(iris_num_cb, iris_num_split):
    X_train, X_test, y_train, y_test = iris_num_split
    energymap(iris_num_cb, X_train)
    return plt.gcf()

@pytest.mark.mpl_image_compare
def test_energymap_prepare_ax(iris_num_cb, iris_num_split):
    X_train, X_test, y_train, y_test = iris_num_split
    ax, transform = prepare_ax(iris_num_cb, X_test)
    energymap(iris_num_cb, transform=transform, ax=ax)
    return plt.gcf()

@pytest.mark.mpl_image_compare
def test_energymap_prepare_ax_scatter(iris_num_cb, iris_num_split):
    X_train, X_test, y_train, y_test = iris_num_split
    ax, transform = prepare_ax(iris_num_cb, X_test)
    energymap(iris_num_cb, transform=transform, ax=ax)
    plot_cb(iris_num_cb, alpha=0.5, transform=transform, ax=ax)
    plot_ref(iris_num_cb, X_test, y_test, iris_num_cb.predict(X_test), alpha=0.5, transform=transform, ax=ax)
    return plt.gcf()