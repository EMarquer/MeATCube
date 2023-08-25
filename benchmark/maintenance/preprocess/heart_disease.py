import pandas as pd
try:
    from . import _utils as utils
except ImportError:
    import _utils as utils

columns = ["#3 (age)","#4 (sex)","#9 (cp)","#10 (trestbps)","#12 (chol)","#16 (fbs)","#19 (restecg)","#32 (thalach)","#38 (exang)","#40 (oldpeak)","#41 (slope)","#44 (ca)","#51 (thal)","#58 (num)"]
target_column = "#58 (num)"
numeric_columns = ["#3 (age)","#10 (trestbps)","#12 (chol)","#16 (fbs)","#32 (thalach)","#40 (oldpeak)","#44 (ca)"]
symbolic_columns = set(columns).difference(numeric_columns + [target_column])

if __name__ == "__main__":
    # multiple .data files
    data_file_name = [
        file_name for file_name in utils.auto_data_path_from_preprocess_file(__file__)
        if "processed.cleveland.data" in file_name
    ][0]
    df = pd.read_csv(data_file_name, header=None)
    df.columns = columns

    # expects 8 missing values
    size_before = len(df)
    df = utils.drop_missing(df)
    print(f"removed {size_before-len(df)}: {size_before} -> {len(df)}")

    # split
    for c in symbolic_columns:
        df[c] = utils.symbolic_to_symbol_ids(df[c])[0]
    X = df[[c for c in columns if c!=target_column]]
    y, y_meaning = utils.symbolic_to_symbol_ids(df[target_column])
    utils.enforce_num(X, numeric_columns)

    # convert to numpy
    np_X, np_y, numeric_columns_idx = utils.pd_to_numpy(X, y, numeric_columns)
    # save
    utils.save_dataset_as_pickle(np_X, np_y, numeric_columns_idx, __file__, y_meaning)
    # test if loading works properly
    print(utils.load_dataset_from_pickle(utils.auto_dataset_name_from_preprocess_file(__file__)))