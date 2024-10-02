from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

USE_STRING_VALUES = True
RUN_EXAMPLE = 0 # 0 for all examples, or example number

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
from meatcube.metrics import f1_score, precision_recall_fscore_support, accuracy
from meatcube.maintenance import decrement_early_stopping

# create the CB
source_similarity = lambda x,y: np.exp(- np.linalg.norm(x - y))
outcome_similarity = lambda x,y: (True if x == y else False)
cb = mc.CB(X_train, y_train, y_values, source_similarity, outcome_similarity)

# performance of the CB
f1, acc = f1_score(cb, X_test, y_test), accuracy(cb, X_test, y_test)
print(f"Initial performance of the CB: F1 {f1}, Accuracy {acc}")

if RUN_EXAMPLE == 0 or RUN_EXAMPLE == 1:
    # --- Example 1 ---
    # repeat removals (1 by 1) until nothing remains in the CB
    print(f"Example 1: Repeat removals until nothing remains in the CB")
    cb_, records, record_final = decrement_early_stopping(cb, X_test, y_test,
                    strategy="hinge",
                    monitor="CB size",
                    register=["F1", "accuracy"],
                    margin=0,
                    step_size=1,
                    return_all=True,
                    patience=len(cb))
    print(f"New performance of the CB: F1 {record_final['F1']}, Accuracy {record_final['accuracy']}, stop at step {record_final['step']} with a CB of size {len(cb_)}")

    # plot evolution of F1 with Seaborn
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame.from_records(records)
    sns.lineplot(data=df[["step", "F1"]], x="step", y="F1")
    plt.show()
    plt.clf()

if RUN_EXAMPLE == 0 or RUN_EXAMPLE == 2:
    # --- Example 2 ---
    # repeat removals until no improvement is observed in the F1 (wait 5 steps before stopping)
    print(f"Example 2: Repeat removals until no improvement is observed in the F1 (wait 5 steps before stopping)")
    cb_, records, record_final = decrement_early_stopping(cb, X_test, y_test,
                    strategy="hinge",
                    monitor="F1",
                    register=["F1", "accuracy"],
                    margin=0,
                    step_size=1,
                    return_all=True,
                    patience=5)
    print(f"New performance of the CB: F1 {record_final['F1']}, Accuracy {record_final['accuracy']}, stop at step {record_final['step']} with a CB of size {len(cb_)}")

    # plot evolution of F1 with Seaborn
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame.from_records(records)
    sns.lineplot(data=df[["step", "F1"]], x="step", y="F1")
    plt.show()
    plt.clf()

if RUN_EXAMPLE == 0 or RUN_EXAMPLE == 3:
    # --- Example 3 ---
    # repeat removals until no improvement larger than 0.05 is observed in the accuracy (wait 2 steps before stopping, remove 5 by 5)
    print(f"Example 3: Repeat removals until no improvement larger than 0.05 is observed in the accuracy (wait 2 steps before stopping, remove 5 cases at each step)")
    cb_, records, record_final = decrement_early_stopping(cb, X_test, y_test,
                    strategy="hinge",
                    monitor="accuracy",
                    register=["F1", "accuracy"],
                    margin=0.05,
                    step_size=5,
                    return_all=True,
                    patience=2)
    print(f"New performance of the CB: F1 {record_final['F1']}, Accuracy {record_final['accuracy']}, stop at step {record_final['step']} with a CB of size {len(cb_)}")

    # plot evolution of F1 with Seaborn
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame.from_records(records)
    sns.lineplot(data=df[["step", "F1"]], x="step", y="F1")
    plt.show()
    plt.clf()

if RUN_EXAMPLE == 0 or RUN_EXAMPLE == 4:
    # --- Example 4 ---
    # repeat removals until no improvement is observed in the hinge competence (wait 1 steps before stopping)
    print(f"Example 4: Repeat removals until no improvement is observed in the hinge competence (wait 1 steps before stopping)")
    cb_, records, record_final = decrement_early_stopping(cb, X_test, y_test,
                    strategy="hinge",
                    monitor="hinge",
                    register=["F1", "accuracy", "hinge"],
                    return_all=True,
                    patience=1)
    print(f"New performance of the CB: F1 {record_final['F1']}, Accuracy {record_final['accuracy']}, stop at step {record_final['step']} with a CB of size {len(cb_)}")

    # plot evolution of F1 with Seaborn
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame.from_records(records)
    sns.lineplot(data=df[["step", "F1"]], x="step", y="F1")
    plt.show()
    plt.clf()