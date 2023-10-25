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

TUNE_SIMILARITY_FOR_KNN = False

# for the weight estimation experiment
RESULT_FOLDER = os.path.join(THIS_FOLDER, "results-no-sim-tuning")
TUNE_SIMILARITY_FOR_MEATCUBE = True
if TUNE_SIMILARITY_FOR_MEATCUBE:
    RESULT_FOLDER = RESULT_FOLDER+"+"

DISPLAY_REPORTED_BASELINE = False

# %%
md_report = f"""
# Benchmark report
TUNE_SIMILARITY_FOR_MEATCUBE: {TUNE_SIMILARITY_FOR_MEATCUBE}
RETUNE_SIMILARITY_FOR_KNN: {TUNE_SIMILARITY_FOR_KNN}
"""

baseline_file = os.path.join(THIS_FOLDER, "baseline.csv")
baseline_df = pd.read_csv(baseline_file, header=0)
for dataset in dataset_utils.DATASETS:
    result_folder_dataset = os.path.join(RESULT_FOLDER, dataset)
    result_file = os.path.join(result_folder_dataset, "summary.pkl")
    result_checkpoint_folder = os.path.join(result_folder_dataset, "checkpoints")
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
        if TUNE_SIMILARITY_FOR_MEATCUBE:
            source_similarity.fit(X, y)
        outcome_similarity = SymbolicSimilarity()

        # create the CB
        cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)
        cb = cb.to(DEVICE)
        cb.compute_sim_matrix()

        # prepare the KNN callback
        def _all_nn_callback(cb: mc.CB, X_, y_):
            try:
                if TUNE_SIMILARITY_FOR_KNN:
                    knn = KNN(MultivaluedSimilarity(numeric_columns, symbolic_columns)).fit(cb.CB_source, cb.CB_outcome)
                else:
                    knn = KNN(source_similarity, cb.CB_source, cb.CB_outcome)
                return accuracy_knn(knn, X_, y_)
            except ValueError:
                return -1
        def _1_nn_callback(cb: mc.CB, X_, y_):
            try:
                if TUNE_SIMILARITY_FOR_KNN:
                    knn = KNN(MultivaluedSimilarity(numeric_columns, symbolic_columns)).fit(cb.CB_source, cb.CB_outcome)
                else:
                    knn = KNN(source_similarity, cb.CB_source, cb.CB_outcome)
                return accuracy_knn(knn, X_, y_, k=1)
            except ValueError:
                return -1

        # compress
        _, records, __ = decrement_early_stopping(cb, X_dev, y_dev, batch_size=8,
            register=["accuracy", "F1", "CB size",
                      ("weighted_nn_accuracy", _all_nn_callback),
                      ("1nn_accuracy", _1_nn_callback)],
            monitor="CB size", return_all=True, tqdm_args={"dynamic_ncols": True}, checkpoint_folder=result_checkpoint_folder)
        
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
    print(f"\tInitial:         step { initial_record['step']:>4}\tMeATCube F1 { initial_record['F1']:>6.2%}, MeATCube acc { initial_record['accuracy']:>6.2%}, KNN acc { initial_record['weighted_nn_accuracy']:>6.2%}, 1-NN { initial_record['1nn_accuracy']:>6.2%}")
    print(f"\tAt best acc:     step {best_acc_record['step']:>4}\tMeATCube F1 {best_acc_record['F1']:>6.2%}, MeATCube acc {best_acc_record['accuracy']:>6.2%}, KNN acc {best_acc_record['weighted_nn_accuracy']:>6.2%}, 1-NN {best_acc_record['1nn_accuracy']:>6.2%}")
    print(f"\t\tDeletion rate: {1-(best_acc_record['cb_size']/initial_record['cb_size']):%} cases removed from the CB: {initial_record['cb_size']} -> {best_acc_record['cb_size']}")
    print(f"\tAt best F1:      step { best_f1_record['step']:>4}\tMeATCube F1 { best_f1_record['F1']:>6.2%}, MeATCube acc { best_f1_record['accuracy']:>6.2%}, KNN acc { best_f1_record['weighted_nn_accuracy']:>6.2%}, 1-NN { best_f1_record['1nn_accuracy']:>6.2%}")
    print(f"\t\tDeletion rate: {1-( best_f1_record['cb_size']/initial_record['cb_size']):%} cases removed from the CB: {initial_record['cb_size']} -> { best_f1_record['cb_size']}")
    print(f"\tAt best KNN acc: step {best_knn_record['step']:>4}\tMeATCube F1 {best_knn_record['F1']:>6.2%}, MeATCube acc {best_knn_record['accuracy']:>6.2%}, KNN acc {best_knn_record['weighted_nn_accuracy']:>6.2%}, 1-NN {best_knn_record['1nn_accuracy']:>6.2%}")
    print(f"\t\tDeletion rate: {1-(best_knn_record['cb_size']/initial_record['cb_size']):%} cases removed from the CB: {initial_record['cb_size']} -> {best_knn_record['cb_size']}")
    print(f"\tAt best 1NN acc: step {best_1nn_record['step']:>4}\tMeATCube F1 {best_1nn_record['F1']:>6.2%}, MeATCube acc {best_1nn_record['accuracy']:>6.2%}, KNN acc {best_knn_record['weighted_nn_accuracy']:>6.2%}, 1-NN {best_1nn_record['1nn_accuracy']:>6.2%}")
    print(f"\t\tDeletion rate: {1-(best_1nn_record['cb_size']/initial_record['cb_size']):%} cases removed from the CB: {initial_record['cb_size']} -> {best_1nn_record['cb_size']}")

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
        'accuracy': "MeATCube",
        'weighted_nn_accuracy': "all-NN w/ Karabulut" if TUNE_SIMILARITY_FOR_KNN else "all-NN",
        '1nn_accuracy': "1-nn w/ Karabulut" if TUNE_SIMILARITY_FOR_KNN else "1-NN"
    })

    plt.rcParams['figure.constrained_layout.use'] = True
    # baseline and plotting
    x_gap = 0.01 * initial_record['cb_size']
    y_gap = 0.01
    baseline_row = baseline_df[baseline_df["dataset"]==dataset]
    sns.relplot(stacked, x="step", y="accuracy", hue="model", style="model", kind="line")
    if DISPLAY_REPORTED_BASELINE:
        plt.axhline(y = baseline_row["initial knn accuracy"].item(), color = 'r', linestyle = '-', label = "baseline k-NN accuracy")
        plt.annotate("Baseline k-NN acc.", (x_gap, baseline_row["initial knn accuracy"].item()+y_gap))
        for baseline_algo in ["ICF", "RC", "CBE"]:
            if not baseline_row[f"deletion rate {baseline_algo}"].isna().item():
                #print(baseline_row[f"deletion rate {baseline_algo}"].item(), initial_record["cb_size"], baseline_row[f"accuracy {baseline_algo}"].item())
                x__ = [float(baseline_row[f"deletion rate {baseline_algo}"].item()) * initial_record["cb_size"]]
                y__ = [float(baseline_row[f"accuracy {baseline_algo}"].item())]
                plt.scatter(x__, y__, color='r')
                plt.annotate(baseline_algo, (x__[0]+x_gap, y__[0]-y_gap))

    for i, (record, label, score) in enumerate([
        (best_f1_record, "Best MeATCube F1", best_f1_record["F1"]),
        (best_acc_record, "Best MeATCube acc.", best_acc_record["accuracy"]),
        (best_knn_record, "Best all-NN acc.", best_knn_record["weighted_nn_accuracy"]),
        (best_1nn_record, "Best 1-NN acc.", best_1nn_record["1nn_accuracy"]),
        ]):
        plt.axvline(x = record["step"], color = 'b', linestyle = ':', alpha=0.5)
        plt.annotate(label + f": {score:.2%}", (record["step"]+x_gap, -(i * 0.04)))
    plt.title("Model accuracy when compressing the CB\nusing MeATCube and hinge competence")

    plt.ylim(-0.2,1.05)
    os.makedirs(os.path.dirname(result_fig_file), exist_ok=True)
    plt.savefig(result_fig_file,  bbox_inches='tight')
    #plt.show()
    plt.clf()

with open("REPORT.md", "w") as f:
    f.write(md_report)

# %%
# step VS time
all_records = []
for dataset in dataset_utils.DATASETS:
    result_file = os.path.join(RESULT_FOLDER, dataset, "summary.pkl")
    with open(result_file, "rb") as f:
        records = pickle.load(f)
        all_records += [
            {**record, "dataset": dataset} for record in records
        ]

df = pd.DataFrame.from_records(all_records)[['cb_size', 'step_time', "dataset"]].dropna()
df["Time per compression step (in seconds)"] = df["step_time"].apply(lambda x: x.total_seconds())
g = sns.lineplot(df, x="cb_size", y="Time per compression step (in seconds)", hue="dataset")
#sns.relplot(df, x="cb_size", y="step_time", hue="dataset", style="dataset", kind="line")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

result_fig_file = os.path.join(RESULT_FOLDER, "figs", "runtime.png")
plt.show()
#plt.savefig(result_fig_file,  bbox_inches='tight')
#plt.clf()

# %%
import numpy as np
import matplotlib.axes
df["$|CB|^2$"] = df["cb_size"] * df["cb_size"]
fig, ax = plt.subplots(figsize=(10, 5))
g: matplotlib.axes.Axes = sns.lineplot(df, ax=ax, x="$|CB|^2$", y="Time per compression step (in seconds)", hue="dataset")
#g.set(yscale='log')

#g.invert_xaxis()
def sqrt(x): return x**(1/2)
def sq(x): return x**(2)

g.set_xlim(0, df["$|CB|^2$"].max()*1.05)
g.set_ylim(0, df["Time per compression step (in seconds)"].max()*1.05)
ax2 = g.secondary_xaxis('top', functions=(sqrt, sq)) # square at bottom, root
ax2.set_xlabel("$|CB|$")

#sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1.3), ncol=3)
result_fig_file = os.path.join(RESULT_FOLDER, "figs", "runtime-square.png")
#plt.show()
plt.savefig(result_fig_file,  bbox_inches='tight')
plt.clf()
# with open("REPORT.md", "w") as f:
#     f.write(md_report)
# %%
# linear regression
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
for dataset in df["dataset"].unique():
    data=df[df["dataset"]==dataset]
    X, y =  data["$|CB|^2$"], data["Time per compression step (in seconds)"]
    lin = linregress(X, y)
    #LinearRegression().fit(X.to_numpy().reshape(-1,1), y.to_numpy().reshape(-1,1))
    print(f"slope: {lin.slope:.3e}", f"pvalue: {lin.pvalue:.3e}", f"rvalue: {lin.rvalue:.3e}", dataset)




# %% add values of K
#K_NN_VALUES = []
K_NN_VALUES = [5, 10]
if K_NN_VALUES:
    for dataset in dataset_utils.DATASETS:
        result_folder_dataset = os.path.join(RESULT_FOLDER, dataset)
        result_file = os.path.join(result_folder_dataset, "summary.pkl")
        result_fig_file = os.path.join(RESULT_FOLDER, "figs", dataset + ".png")
        result_fig_smooth_file = os.path.join(RESULT_FOLDER, "figs", dataset + "-smooth.png")
        knn_result_file = os.path.join(result_folder_dataset, "summary_with_knns.pkl")
        result_checkpoint_folder = os.path.join(result_folder_dataset, "checkpoints")
        
        if RECOMPUTE or not os.path.exists(knn_result_file):
            with open(result_file, "rb") as f:
                records = pickle.load(f)
            state_dict = dataset_utils.load_dataset_from_pickle(dataset)
            X = state_dict["X"]
            y = state_dict["y"]
            y_values = np.unique(y)
            numeric_columns = state_dict["numeric_columns"]
            symbolic_columns = state_dict["symbolic_columns"]
            
            from sklearn.model_selection import KFold, train_test_split
            X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=.2, random_state=42)
            X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=.25, random_state=42)

            source_similarity = MultivaluedSimilarity(numeric_columns, symbolic_columns)
            if TUNE_SIMILARITY_FOR_MEATCUBE:
                source_similarity.fit(X, y)

            max_cb_size = X_train.shape[0]
            updated_records = []
            from tqdm import tqdm
            for record in tqdm(records):
                # load the cb
                checkpoint_file = os.path.join(result_checkpoint_folder, 
                    f"cb_step_{str(record['step']).rjust(len(str(max_cb_size)), '0')}.pkl")
                with open(checkpoint_file, 'rb') as f:
                    cb = pickle.load(f)

                # add each K-NN result to the record
                updated_record = {**record}
                for k in K_NN_VALUES: 
                    knn = KNN(source_similarity, cb.CB_source, cb.CB_outcome)
                    updated_record[f"{k}nn_accuracy"] = accuracy_knn(knn, X_dev, y_dev, k=k)
                updated_records.append(updated_record)
            #print(updated_records)
            
            with open(knn_result_file, "wb") as f:
                pickle.dump(updated_records, f)
        else:
            with open(knn_result_file, "rb") as f:
                updated_records = pickle.load(f)

        ## %%
        df = pd.DataFrame.from_records(updated_records)[['step', 'accuracy', 'weighted_nn_accuracy', '1nn_accuracy', *[f"{k}nn_accuracy" for k in K_NN_VALUES]]]
        stacked = (df.set_index('step').stack().reset_index())
        stacked.columns = ["step", "model", "accuracy"]
        stacked["accuracy"][stacked["accuracy"] < 0] = pd.NA
        stacked=stacked.dropna(axis=0)
        stacked["model"] = stacked["model"].replace({
            'accuracy': "MeATCube",
            'weighted_nn_accuracy': "all-NN w/ Karabulut" if TUNE_SIMILARITY_FOR_KNN else "all-NN",
            '1nn_accuracy': "1-nn w/ Karabulut" if TUNE_SIMILARITY_FOR_KNN else "1-NN",
            **{f"{k}nn_accuracy": f"{k}-NN" for k in K_NN_VALUES}
        })
        hue_order = [
            "MeATCube",
            ("1-nn w/ Karabulut" if TUNE_SIMILARITY_FOR_KNN else "1-NN"),
            *[f"{k}-NN" for k in K_NN_VALUES],
            ("all-NN w/ Karabulut" if TUNE_SIMILARITY_FOR_KNN else "all-NN"),
        ]
        # compute smoothed scores
        from scipy.ndimage.filters import gaussian_filter1d
        stacked_smooth = stacked.copy()
        for model in stacked["model"].unique():
            mask = stacked_smooth["model"] == model
            stacked_smooth.loc[mask,"accuracy"] = gaussian_filter1d(
                stacked_smooth.loc[mask]["accuracy"],
                sigma=max(1,np.floor(((mask).sum()*0.025)))
            )


        plt.rcParams['figure.constrained_layout.use'] = True
        # baseline and plotting
        x_gap = 0.01 * initial_record['cb_size']
        y_gap = 0.01
        baseline_row = baseline_df[baseline_df["dataset"]==dataset]
        fig, ax = plt.subplots(figsize=(10, 5))
        g: matplotlib.axes.Axes=sns.lineplot(stacked, ax=ax, x="step", y="accuracy", hue="model",
                      hue_order=hue_order,
                      style_order=hue_order,
                      style="model", alpha=0.3,
                      legend=False)
        # stacked_smooth = stacked_smooth.sort_values("model",
        #     key = np.vectorize(lambda x: hue_order.index(x)))
        g: matplotlib.axes.Axes=sns.lineplot(stacked_smooth,
                    ax=g, x="step", y="accuracy", hue="model",
                      hue_order=hue_order,
                      style_order=hue_order,
                      style="model")
        hue_style = {model: (line.get_color(), line.get_linestyle())
                for model, line in zip(hue_order, g.get_legend().get_lines())}
        
        sns.move_legend(g, "lower left", bbox_to_anchor=(1, 0),
            title=f"Dataset:\n  {dataset}"
            "\n\nVertical lines:\n  Highest model performance"
            "\n\nLine opacity:\n  Light: actual values\n  Dark: smoothed values"
            "\n\nModels:")
        g.spines['top'].set_visible(False)
        g.spines['right'].set_visible(False)
        
        if DISPLAY_REPORTED_BASELINE:
            plt.axhline(y = baseline_row["initial knn accuracy"].item(), color = 'r', linestyle = '-', label = "baseline k-NN accuracy")
            plt.annotate("Baseline k-NN acc.", (x_gap, baseline_row["initial knn accuracy"].item()+y_gap))
            for baseline_algo in ["ICF", "RC", "CBE"]:
                if not baseline_row[f"deletion rate {baseline_algo}"].isna().item():
                    #print(baseline_row[f"deletion rate {baseline_algo}"].item(), initial_record["cb_size"], baseline_row[f"accuracy {baseline_algo}"].item())
                    x__ = [float(baseline_row[f"deletion rate {baseline_algo}"].item()) * initial_record["cb_size"]]
                    y__ = [float(baseline_row[f"accuracy {baseline_algo}"].item())]
                    plt.scatter(x__, y__, color='r')
                    plt.annotate(baseline_algo, (x__[0]+x_gap, y__[0]-y_gap))

        best_f1_record  = sorted(updated_records, key=lambda record: (record["F1"], record["step"]), reverse=True)[0]
        best_acc_record = sorted(updated_records, key=lambda record: (record["accuracy"], record["step"]), reverse=True)[0]
        best_allnn_record = sorted(updated_records, key=lambda record: (record["weighted_nn_accuracy"], record["step"]), reverse=True)[0]
        best_1nn_record = sorted(updated_records, key=lambda record: (record["1nn_accuracy"], record["step"]), reverse=True)[0]
        best_knn_records = {k:
            sorted(updated_records, key=lambda record: (record[f"{k}nn_accuracy"], record["step"]), reverse=True)[0]
            for k in K_NN_VALUES
        }
        record_label_score_line_list = [
            #(best_f1_record, "Best MeATCube F1", best_f1_record["F1"], hue_style["MeATCube"]),
            (best_acc_record, "Best MeATCube acc.", best_acc_record["accuracy"], hue_style["MeATCube"]),
            (best_1nn_record, "Best 1-NN acc.", best_1nn_record["1nn_accuracy"], hue_style["1-nn w/ Karabulut" if TUNE_SIMILARITY_FOR_KNN else "1-NN"]),
            *[
                (best_knn_records[k], f"Best {k}-NN acc.", best_knn_records[k][f"{k}nn_accuracy"], hue_style[f"{k}-NN"]) for k in K_NN_VALUES
             ],
            (best_allnn_record, "Best all-NN acc.", best_allnn_record["weighted_nn_accuracy"], hue_style["all-NN w/ Karabulut" if TUNE_SIMILARITY_FOR_KNN else "all-NN"]),
        ]

        y_space = 0.05
        y_max = (1+y_space+(len(record_label_score_line_list) * y_space))
        relative_100_line = 1/y_max
        for i, (record, label, score, (color, style)) in enumerate(record_label_score_line_list):
            i = len(record_label_score_line_list)-1-i
            height_bot_line = (1+y_space+(i * y_space))
            relative_height_bot_line = height_bot_line/y_max
            relative_height_top_line = (height_bot_line+(y_space*0.9))/y_max
            #plt.axvline(x = record["step"], ymin=0, ymax=relative_100_line, color = color, linestyle = ':', alpha=0.2)
            plt.axvline(x = record["step"], ymin=0, ymax=relative_height_bot_line, color = color, linestyle = ':', alpha=0.2)
            #plt.axvline(x = record["step"], ymin=relative_100_line, ymax=1, color = color, linestyle = ':', alpha=0.8)
            plt.axvline(x = record["step"], ymin=relative_height_bot_line, ymax=relative_height_top_line, color = color, alpha=0.8)
            text = plt.annotate(label + f": {score:.2%}", (record["step"]+(x_gap*2), height_bot_line+y_gap))
            #text.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
            #plt.annotate(label + f": {score:.2%}", (record["step"]+x_gap, 0.02-(i * 0.04)))
        plt.title(f"Model accuracy when compressing the CB,\nusing MeATCube and hinge competence")
        
        #plt.ylim(-0.2,1.05)
        plt.ylim(0,y_max)
        ticks_values, ticks_labels = plt.yticks()
        ticks_values = [ticks_value for ticks_value in ticks_values if ticks_value<=1]
        ticks_labels = [ticks_label for ticks_label in ticks_labels if ticks_label.get_position()[1]<=1]
        for value, label in zip(ticks_values, ticks_labels):
            label.set_text(f'{value:.0%}')
        plt.yticks(ticks_values, ticks_labels)
        os.makedirs(os.path.dirname(result_fig_file), exist_ok=True)
        plt.savefig(result_fig_file,  bbox_inches='tight')
        #plt.show(); break
        plt.clf()

# %%
