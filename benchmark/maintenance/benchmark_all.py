# %%
import pandas as pd
import numpy as np
import torch
import sys, os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sim import MultivaluedSimilarity, SymbolicSimilarity
from knn import KNN, accuracy as accuracy_knn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import meatcube as mc
from meatcube.maintenance import decrement_early_stopping, decrement
from meatcube.metrics import accuracy
sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocess/'))
import _utils as dataset_utils

# refers to https://cora.ucc.ie/server/api/core/bitstreams/39193798-3fe0-461a-b1b6-3d9cffd108d3/content
THIS_FOLDER = os.path.dirname(__file__)
RESULT_FOLDER = os.path.join(THIS_FOLDER, "results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RECOMPUTE = False

md_report = """
# Benchmark report
"""

baseline_file = os.path.join(THIS_FOLDER, "baseline.csv")
baseline_df = pd.read_csv(baseline_file, header=0)
for dataset in dataset_utils.DATASETS:
    result_file = os.path.join(RESULT_FOLDER, dataset + ".pkl")
    result_fig_file = os.path.join(RESULT_FOLDER, "figs", dataset + ".png")
    if RECOMPUTE or not os.path.exists(result_file):
        print(f"{dataset} dataset preparation ---")
        # load the data
        state_dict = dataset_utils.load_dataset_from_pickle(dataset)
        X = state_dict["X"]
        y = state_dict["y"]
        y_values = np.unique(y)
        numeric_columns = state_dict["numeric_columns"]
        symbolic_columns = state_dict["symbolic_columns"]

        # 10 splits of 60%, 20%, with 20% test set
        from sklearn.model_selection import KFold, train_test_split
        X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=.25, random_state=42)

        # setting up the similarity
        source_similarity = MultivaluedSimilarity(numeric_columns, symbolic_columns)
        #source_similarity.fit(X, y)
        outcome_similarity = SymbolicSimilarity()

        # create the CB
        cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)
        cb = cb.to(DEVICE)
        cb.compute_sim_matrix()

        # prepare the KNN callback
        def knn_callback(cb: mc.CB, X_, y_):
            try:
                knn = KNN(MultivaluedSimilarity(numeric_columns, symbolic_columns)).fit(cb.CB_source, cb.CB_outcome)
                return accuracy_knn(knn, X_, y_)
            except ValueError:
                return -1
        def _1nn_callback(cb: mc.CB, X_, y_):
            try:
                knn = KNN(MultivaluedSimilarity(numeric_columns, symbolic_columns)).fit(cb.CB_source, cb.CB_outcome)
                return accuracy_knn(knn, X_, y_, k=1)
            except ValueError:
                return -1

        # compress
        _, records, __ = decrement_early_stopping(cb, X_dev, y_dev, batch_size=8,
            register=["accuracy", "F1", "CB size", ("weighted_nn_accuracy", knn_callback), ("1nn_accuracy", _1nn_callback)],
            monitor="CB size", return_all=True, tqdm_args={"dynamic_ncols": True})
        
        # save the run
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "wb") as f:
            pickle.dump(records, f)
        
    else:
        with open(result_file, "rb") as f:
            records = pickle.load(f)

    # performance
    initial_record = records[0]
    best_f1_record  = sorted(records, key=lambda record: (record["F1"], record["step"]), reverse=True)[0]
    best_acc_record = sorted(records, key=lambda record: (record["accuracy"], record["step"]), reverse=True)[0]
    best_knn_record = sorted(records, key=lambda record: (record["weighted_nn_accuracy"], record["step"]), reverse=True)[0]
    best_1nn_record = sorted(records, key=lambda record: (record["1nn_accuracy"], record["step"]), reverse=True)[0]
    print(f"{dataset} dataset ---")
    # print(f"\tInitial:         step { initial_record['step']:>4}\tMeATCube F1 { initial_record['F1']:>6.2%}, MeATCube acc { initial_record['accuracy']:>6.2%}, KNN acc { initial_record['weighted_nn_accuracy']:>6.2%}, 1-NN { initial_record['1nn_accuracy']:>6.2%}")
    # print(f"\tAt best acc:     step {best_acc_record['step']:>4}\tMeATCube F1 {best_acc_record['F1']:>6.2%}, MeATCube acc {best_acc_record['accuracy']:>6.2%}, KNN acc {best_acc_record['weighted_nn_accuracy']:>6.2%}, 1-NN {best_acc_record['1nn_accuracy']:>6.2%}")
    # print(f"\t\tDeletion rate: {1-(best_acc_record['cb_size']/initial_record['cb_size']):%} cases removed from the CB: {initial_record['cb_size']} -> {best_acc_record['cb_size']}")
    # print(f"\tAt best F1:      step { best_f1_record['step']:>4}\tMeATCube F1 { best_f1_record['F1']:>6.2%}, MeATCube acc { best_f1_record['accuracy']:>6.2%}, KNN acc { best_f1_record['weighted_nn_accuracy']:>6.2%}, 1-NN { best_f1_record['1nn_accuracy']:>6.2%}")
    # print(f"\t\tDeletion rate: {1-( best_f1_record['cb_size']/initial_record['cb_size']):%} cases removed from the CB: {initial_record['cb_size']} -> { best_f1_record['cb_size']}")
    # print(f"\tAt best KNN acc: step {best_knn_record['step']:>4}\tMeATCube F1 {best_knn_record['F1']:>6.2%}, MeATCube acc {best_knn_record['accuracy']:>6.2%}, KNN acc {best_knn_record['weighted_nn_accuracy']:>6.2%}, 1-NN {best_knn_record['1nn_accuracy']:>6.2%}")
    # print(f"\t\tDeletion rate: {1-(best_knn_record['cb_size']/initial_record['cb_size']):%} cases removed from the CB: {initial_record['cb_size']} -> {best_knn_record['cb_size']}")
    # print(f"\tAt best 1NN acc: step {best_1nn_record['step']:>4}\tMeATCube F1 {best_1nn_record['F1']:>6.2%}, MeATCube acc {best_1nn_record['accuracy']:>6.2%}, KNN acc {best_knn_record['weighted_nn_accuracy']:>6.2%}, 1-NN {best_1nn_record['1nn_accuracy']:>6.2%}")
    # print(f"\t\tDeletion rate: {1-(best_1nn_record['cb_size']/initial_record['cb_size']):%} cases removed from the CB: {initial_record['cb_size']} -> {best_1nn_record['cb_size']}")

    df = pd.DataFrame.from_records([initial_record, best_f1_record, best_acc_record, best_knn_record, best_1nn_record])
    df.index = [
        "initial",
        "best MeATCube F1",
        "best MeATCube acc.",
        "best KNN acc.",
        "best 1NN acc."]
    df["deletion rate"] = 1 - (df['cb_size']/initial_record['cb_size'])
    df_ = df[["deletion rate", 'cb_size', 'step', 'F1', 'accuracy', 'weighted_nn_accuracy', '1nn_accuracy']]
    for c in ["deletion rate",'F1','accuracy','weighted_nn_accuracy','1nn_accuracy']:
        df_[c] = df_[c].apply(lambda x: f"{x:.2%}")
    #print(df[["deletion rate", 'cb_size', 'step', 'F1', 'accuracy', 'weighted_nn_accuracy', '1nn_accuracy']])
    md_report += f"\n\n## {dataset} dataset\n" + df_.to_markdown()
    md_report += f"\n\n![](results/figs/{dataset}.png)\n"
    print(df_.to_markdown())

    df = pd.DataFrame.from_records(records)[['step', 'accuracy', 'weighted_nn_accuracy', '1nn_accuracy']]
    stacked = (df.set_index('step').stack().reset_index())
    stacked.columns = ["step", "model", "accuracy"]
    stacked["accuracy"][stacked["accuracy"] < 0] = pd.NA
    stacked=stacked.dropna(axis=0)
    stacked["model"] = stacked["model"].replace({
        'accuracy': "MeATCube", 'weighted_nn_accuracy': "k-NN", '1nn_accuracy': "1-nn"
    })
    baseline_row = baseline_df[baseline_df["dataset"]==dataset]
    sns.relplot(stacked, x="step", y="accuracy", hue="model", style="model", kind="line")
    plt.axhline(y = baseline_row["initial knn accuracy"].item(), color = 'r', linestyle = '-', label = "baseline k-NN accuracy")
    for baseline_algo in ["ICF", "RC", "CBE"]:
        if not baseline_row[f"deletion rate {baseline_algo}"].isna().item():
            #print(baseline_row[f"deletion rate {baseline_algo}"].item(), initial_record["cb_size"], baseline_row[f"accuracy {baseline_algo}"].item())
            x__ = [float(baseline_row[f"deletion rate {baseline_algo}"].item()) * initial_record["cb_size"]]
            y__ = [float(baseline_row[f"accuracy {baseline_algo}"].item())]
            plt.scatter(x__, y__)
            plt.annotate(baseline_algo, (x__[0], y__[0]))

    plt.ylim(0,1.05)
    os.makedirs(os.path.dirname(result_fig_file), exist_ok=True)
    plt.savefig(result_fig_file)
    plt.clf()

with open("REPORT.md", "w") as f:
    f.write(md_report)
# %%
