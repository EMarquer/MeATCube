from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm
from typing import List, Literal, Tuple, Any, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.animation
FuncAnimation = matplotlib.animation.FuncAnimation

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from ..models.AbstractEnergyBasedClassifier import ACaseBaseEnergyClassifier
from ..cb_maintenance import CBClassificationMaintainer

def split_datasets(datasets: List, require_ref_set=False, random_state=42, ref_size=0.4, test_size=0.3):
    datasets_split = []
    for dataset in datasets:
        X, y = dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        if require_ref_set:
            X_train, X_ref, y_train, y_ref = train_test_split(
                X_train, y_train, test_size=ref_size, random_state=random_state
            )
            datasets_split.append((
                (X_train, y_train),
                (X_ref, y_ref),
                (X_test, y_test),
                ))

        else:
            datasets_split.append((
                (X_train, y_train),
                (X_test, y_test),
                ))
            
    return datasets_split

def plot_dataset_model_grid_autosplit(datasets: List,
                            dataset_names: List[str],
                            classifiers: List,
                            classifiers_names: List[str],
                            dataset_progress_names: List[str]=None,
                            classifiers_progress_names: List[str]=None,
                            progress_bar: Literal[True, "notebook", False]=True,
                            figure:plt.Figure=None,
                            axes: plt.Axes=None,
                            fit_estimators=True,
                            DecisionBoundaryDisplay_kwargs=None) -> Tuple[plt.Figure, plt.Axes]:
    
    require_ref_set = any(isinstance(clf, CBClassificationMaintainer) for clf in classifiers)
    datasets_split = split_datasets(datasets, require_ref_set)
    

    return plot_dataset_model_grid(
        datasets_split,
        dataset_names,
        classifiers,
        classifiers_names,
        dataset_progress_names,
        classifiers_progress_names,
        progress_bar,
        figure,
        axes,
        fit_estimators,
        DecisionBoundaryDisplay_kwargs)


def plot_dataset_model_grid(datasets_split: Union[List[Tuple[Any, Any]], List[Tuple[Any, Any, Any]]],
                            dataset_names: List[str],
                            classifiers: List,
                            classifiers_names: List[str],
                            dataset_progress_names: List[str]=None,
                            classifiers_progress_names: List[str]=None,
                            progress_bar: Literal[True, "notebook", False]=True,
                            figure:plt.Figure=None,
                            axes: plt.Axes=None,
                            fit_estimators=True,
                            DecisionBoundaryDisplay_kwargs=None) -> Tuple[plt.Figure, plt.Axes]:
    """_summary_

    Parameters
    ----------
    datasets : Union[List[Tuple[Any, Any]], List[Tuple[Any, Any, Any]]]
        A list of pairs ((X_train, y_train), (X_test, y_test)) or triplet ((X_train, y_train), (X_ref, y_ref), (X_test, y_test))
        to use for the prediction algorithms.
    dataset_names : List[str]
        _description_
    classifiers : List
        _description_
    classifiers_names : List[str]
        _description_
    dataset_progress_names : List[str], optional
        _description_, by default None
    classifiers_progress_names : List[str], optional
        _description_, by default None
    progress_bar : Literal[True, &quot;notebook&quot;, False], optional
        _description_, by default True
    figure : plt.Figure, optional
        _description_, by default None
    axes : plt.Axes, optional
        _description_, by default None
    fit_estimators : bool, optional
        _description_, by default True
    DecisionBoundaryDisplay_kwargs : _type_, optional
        _description_, by default None

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        _description_
    """
    # check if there are any untrained estimators but fit is not asked
    if not fit_estimators:
        for clf in classifiers:
            if isinstance(clf, ClassifierMixin):
                check_is_fitted(clf)

    # check if there is maintenance planed, and if there is make sure every dataset is equipped with train-dev-test sets
    if any(isinstance(clf, CBClassificationMaintainer) for clf in classifiers):
        for dataset, dataset_name in zip(datasets_split, dataset_name):
            assert len(dataset) >= 3, f"Dataset '{dataset_name}' does not contain a train/dev/test split, however the dev set is required for case base maintenance"


    # prepare decision boundary kwargs
    if DecisionBoundaryDisplay_kwargs is None: 
        DecisionBoundaryDisplay_kwargs = {"alpha":0.5, "grid_resolution":50}
    else:
        DecisionBoundaryDisplay_kwargs = {"alpha":0.5, "grid_resolution":50, **DecisionBoundaryDisplay_kwargs}
    if progress_bar == "notebook":
        tqdm_ = tqdm_notebook
    else:
        tqdm_ = tqdm

    if not dataset_progress_names: dataset_progress_names = dataset_names
    if not classifiers_progress_names: classifiers_progress_names = classifiers_names

    if figure is None and axes is None:
        figure, axes = plt.subplots(figsize=(3*(len(classifiers)+1), 3*len(datasets_split)))

    elif figure is None:
        figure = axes.get_figure()
        
    with tqdm_(total = len(datasets_split)*len(classifiers), disable=True if progress_bar is False else False) as progress:
        i = 1

        # iterate over datasets
        for ds_cnt, (ds, ds_name, ds_progress_name) in enumerate(zip(datasets_split, dataset_names, dataset_progress_names)):  
            progress.set_description(ds_name)

            # preprocess dataset, split into training and test part
            if len(ds) == 2:
                (X_train, y_train), (X_test, y_test) = ds
                X_ref, y_ref = None, None
                X, y = np.concatenate([X_train, X_test], axis=0), np.concatenate([y_train, y_test], axis=0)
            else:
                (X_train, y_train), (X_ref, y_ref), (X_test, y_test), = ds
                X, y = np.concatenate([X_train, X_ref, X_test], axis=0), np.concatenate([y_train, y_ref, y_test], axis=0)

            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

            # just plot the dataset first
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(["#FF0000", "#0000FF"])
            ax = figure.add_subplot(len(datasets_split), len(classifiers) + 1, i)
            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k",
                    label="Training data")
            # Plot the ref points
            if X_ref is not None and y_ref is not None:
                ax.scatter(
                    X_ref[:, 0], X_ref[:, 1], c=y_ref, cmap=cm_bright, alpha=0.6, #edgecolors="k",
                        marker="^",
                        label="Ref data"
                )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, #edgecolors="k",
                    marker="v",
                    label="Test data"
            )
            if ds_cnt == 0:
                ax.set_title("Input data: " + ds_name)
                ax.legend()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            i += 1



            # iterate over classifiers
            for clf, name, progress_name in zip(classifiers, classifiers_names, classifiers_progress_names):
                progress.update()
                progress.set_description(ds_name + ", clf: " + progress_name)
                ax = figure.add_subplot(len(datasets_split), len(classifiers) + 1, i)

                # scaler = StandardScaler()
                # scaler.fit(X_train, y_train)
                # X_train = scaler.transform(X_train)
                # X_test = scaler.transform(X_test)
                # X_test = scaler.transform(X_test)

                if isinstance(clf, CBClassificationMaintainer):
                    if fit_estimators:
                        clf.fit(X_train, y_train, X_ref=X_ref, y_ref=y_ref, warm_start=False, increment_init=0.1)
                    message_bonus = f" |CB|: {clf.initial_estimator_len_} -> {len(clf)}"
                elif isinstance(clf, ACaseBaseEnergyClassifier):
                    if fit_estimators:
                        clf.fit(X_train, y_train)
                    message_bonus = f" |CB|: {len(clf)}"
                elif isinstance(clf, ClassifierMixin):
                    if fit_estimators:
                        clf.fit(X_train, y_train)
                    message_bonus = f""
                else:
                    raise ValueError(f"estimator {clf} must be a ClassifierMixin")
                

                #clf_ = make_pipeline(scaler, clf)
                
                score = clf.score(X_test, y_test)
                DecisionBoundaryDisplay.from_estimator(
                    clf, X, cmap=cm, ax=ax, eps=0.5, **DecisionBoundaryDisplay_kwargs
                )

                # Plot the testing points
                scatter = ax.scatter(
                    X_test[:, 0],
                    X_test[:, 1],
                    c=y_test,
                    cmap=cm_bright,
                    #edgecolors="k",
                    alpha=0.6,
                    marker="v",
                    label="Test data"
                )
                # Plot the ref points
                if X_ref is not None and y_ref is not None:
                    ax.scatter(
                        X_ref[:, 0], X_ref[:, 1], c=y_ref, cmap=cm_bright, alpha=0.6, #edgecolors="k",
                            marker="^",
                            label="Ref data"
                    )

                # Plot the CB
                if isinstance(clf, ACaseBaseEnergyClassifier):
                    try:
                        _X = clf.steps[0][1].inverse_transform(clf._X)
                    except AttributeError:
                        _X = clf._X
                    _y = clf._y
                    scatter = ax.scatter(
                        x=_X[:, 0],
                        y=_X[:, 1],
                        c=_y,
                        cmap=cm_bright,
                        edgecolors="k",
                        alpha=0.6,
                        marker="o",
                        label="CB"
                    )

                # display additional information

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xticks(())
                ax.set_yticks(())
                if ds_cnt == 0:
                    ax.set_title(name)
                ax.text(
                    x_max - 0.3,
                    y_min + 0.3,
                    ("%.2f" % score).lstrip("0") + message_bonus,
                    size=15,
                    horizontalalignment="right",
                )
                
                # produce a legend with a cross-section of sizes from the scatter
                if isinstance(clf, ACaseBaseEnergyClassifier):
                    legend_elements = [
                        plt.Line2D([0], [0], marker='o', color='black', label='CB', markersize=10, linestyle='None'),
                        plt.Line2D([0], [0], marker='v', color='black', label='Test', markersize=10, linestyle='None')
                    ]
                else:
                    legend_elements = [
                        plt.Line2D([0], [0], marker='v', color='black', label='Test', markersize=10, linestyle='None')
                    ]
                if X_ref is not None and y_ref is not None:
                    legend_elements.insert(-1,
                                           plt.Line2D([0], [0], marker='^', color='black', label='Ref', markersize=10, linestyle='None'),
                    )
                ax.legend(handles=legend_elements, title="Data")
                i += 1

    return figure, axes

def animate_dataset_model_grid_on_models_autosplit(
        datasets: List,
        dataset_names: List[str],
        classifiers: List,
        classifiers_names: List[str],
        dataset_progress_names: List[str]=None,
        classifiers_progress_names: List[str]=None,
        progress_bar: Literal[True, "notebook", False]=True,
        figure: plt.Figure=None,
        axes: plt.Axes=None,
        fit_estimators=True,
        DecisionBoundaryDisplay_kwargs=None) -> Tuple[FuncAnimation, plt.Figure , plt.Axes]:

    require_ref_set = any(isinstance(clf, CBClassificationMaintainer) for clf in classifiers)
    datasets_split = split_datasets(datasets, require_ref_set)
    
    return animate_dataset_model_grid_on_models(
            datasets_split,
            dataset_names,
            classifiers,
            classifiers_names,
            dataset_progress_names,
            classifiers_progress_names,
            progress_bar,
            figure,
            axes,
            fit_estimators,
            DecisionBoundaryDisplay_kwargs)

def animate_dataset_model_grid_on_models(
        datasets_split: List,
        dataset_names: List[str],
        classifiers: List,
        classifiers_names: List[str],
        dataset_progress_names: List[str]=None,
        classifiers_progress_names: List[str]=None,
        progress_bar: Literal[True, "notebook", False]=True,
        figure: plt.Figure=None,
        axes: plt.Axes=None,
        fit_estimators=True,
        DecisionBoundaryDisplay_kwargs=None) -> Tuple[FuncAnimation, plt.Figure , plt.Axes]:

    if figure is None and axes is None:
        figure, axes = plt.subplots(figsize=(3*2, 3*len(datasets_split)))
    #fig , ax = plot_dataset_model_grid(datasets, dataset_names, [classifiers[0]], [names[0]], progress_bar="notebook")

    def init():
        axes.clear()
        figure.clear()
    def update(t):
        axes.clear()
        figure.clear()
        
        # optionally clear axes and reset limits
        plot_dataset_model_grid(
            datasets_split,
            dataset_names,
            classifiers[t:t+1],
            classifiers_names[t:t+1],
            dataset_progress_names,
            classifiers_progress_names[t:t+1] if classifiers_progress_names is not None else None,
            progress_bar,
            figure,
            axes,
            fit_estimators,
            DecisionBoundaryDisplay_kwargs)
        plt.tight_layout()

    ani = FuncAnimation(figure, update, frames=range(len(classifiers)), init_func=init)
    return ani, figure, axes