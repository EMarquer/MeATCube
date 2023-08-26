import pandas as pd
try:
    from . import _utils as utils
except ImportError:
    import _utils as utils

columns = ["Class Name", "Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"]
target_column = "Class Name"
numeric_columns = ["Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"]

if __name__ == "__main__":
    data_file_name = utils.auto_data_path_from_preprocess_file(__file__)[0]
    df = pd.read_csv(data_file_name, header=None)
    df.columns = columns
    X = df[[c for c in columns if c!=target_column]]
    y, y_meaning = utils.symbolic_to_symbol_ids(df[target_column])
    
    # convert to numpy
    utils.enforce_num(X, numeric_columns)
    np_X, np_y, numeric_columns_idx = utils.pd_to_numpy(X, y, numeric_columns)
    # save
    utils.save_dataset_as_pickle(np_X, np_y, numeric_columns_idx, __file__, y_meaning)
    # test if loading works properly
    print(utils.load_dataset_from_pickle(utils.auto_dataset_name_from_preprocess_file(__file__)))