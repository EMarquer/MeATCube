import torch, numpy as np
import pandas as pd

def to_numpy_array(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.numpy()
    elif isinstance(values, (pd.DataFrame, pd.Series)):
        return values.to_numpy()
    elif isinstance(values, np.ndarray):
        return values
    else:
        try: # float array-like
            return np.array(values, dtype=float)
        except ValueError: # non-float array-like
            return np.array(values)