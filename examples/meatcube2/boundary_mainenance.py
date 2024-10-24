"""Adapted from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.linalg import norm

from scipy.spatial.distance import euclidean, cosine, correlation, braycurtis, hamming
def euclidean_sim(x1,x2):
    return np.exp(-euclidean(x1,x2))
def cosine_sim(x1,x2):
    return np.exp(-cosine(x1,x2))
def correlation_sim(x1,x2):
    return np.exp(-correlation(x1,x2))
def hamming_sim(x1,x2):
    return np.exp(-hamming(x1,x2))
def radius_sim(x1,x2):
    return np.exp(-abs(norm(x1)-norm(x2)))
def class_equality_sim(y1,y2):
    return np.equal(y1,y2).astype(float)

from meatcube2.models.meatcube_torch import MeATCubeCB
from meatcube2.cb_maintenance import CBClassificationMaintainer
from meatcube2.models.ct_coat import CtCoAT
from meatcube2.models.ct_coat_naive import CtCoAT as CtCoATNaive


names = [
    "Euclidean MeATCube",
    "Euclidean MeATCube increment",
    "Euclidean MeATCube decrement",
    "CtCoat",
    "CtCoat increment",
    "CtCoat decrement",
    # "Cosine MeATCube",
    # "Radius MeATCube",
    # "Correlation MeATCube",
    # "Hamming MeATCube",
    #"CtCoat non-normalized",
    "Nearest Neighbors",
    "Linear SVM",
    # "RBF SVM",
    # "Gaussian Process",
    # "Decision Tree",
]

classifiers = [
    MeATCubeCB(euclidean_sim, class_equality_sim),
    CBClassificationMaintainer(MeATCubeCB(euclidean_sim, class_equality_sim), mode="increment", patience=10, random_state=42),
    CBClassificationMaintainer(MeATCubeCB(euclidean_sim, class_equality_sim), mode="decrement", patience=10, random_state=42),
    CtCoAT(euclidean_sim, class_equality_sim),
    CBClassificationMaintainer(CtCoAT(euclidean_sim, class_equality_sim), mode="increment", patience=10, random_state=42),
    CBClassificationMaintainer(CtCoAT(euclidean_sim, class_equality_sim), mode="decrement", patience=10, random_state=42),
    # MeATCubeCB(cosine_sim, class_equality_sim),
    # MeATCubeCB(radius_sim, class_equality_sim),
    # MeATCubeCB(correlation_sim, class_equality_sim),
    # MeATCubeCB(hamming_sim, class_equality_sim),
    #CtCoAT(euclidean_sim, class_equality_sim, normalize=False),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    # SVC(gamma=2, C=1, random_state=42),
    # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    # DecisionTreeClassifier(max_depth=5, random_state=42),
]
N_SAMPLES = 80#0

X, y = make_classification(n_samples=N_SAMPLES, 
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_moons(N_SAMPLES, noise=0.3, random_state=0),
    make_circles(N_SAMPLES, noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    print(ds_cnt/len(datasets))
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_train, X_ref, y_train, y_ref = train_test_split(
        X_train, y_train, test_size=0.4, random_state=42, stratify=y_train
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        #clf = make_pipeline(StandardScaler(), clf)
        #clf.fit(X_train, y_train)
        if isinstance(clf, CBClassificationMaintainer):
            # print(len(clf))
            #clf_ = CBClassificationMaintainer(clf)
            clf = clf.fit(X_train, y_train, X_ref=X_ref, y_ref=y_ref, warm_start=False, increment_init=0.1)
            #clf = clf_.best_estimator_
            message_bonus = f" |CB|: {clf.initial_estimator_len_} -> {len(clf)}"
            # clf_ = CBClassificationMaintainer(clf, mode="increment")
            # clf_ = clf_.fit(X_train, y_train, X_ref=X_ref, y_ref=y_ref, warm_start=False)
            #clf = clf_.best_estimator_
            print(len(clf))
        elif isinstance(clf, MeATCubeCB):
            clf.fit(X_train, y_train)
            message_bonus = f" |CB|: {len(clf)}"
        else:
            clf.fit(X_train, y_train)
            message_bonus = f""

        score = clf.score(X_test, y_test)
        #QuantileTransformer(n_quantiles=10).fit_transform(score)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.5, ax=ax, eps=0.5, #grid_resolution=500
        )

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            (f"{score:.2f}".lstrip("0")+message_bonus),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.show()