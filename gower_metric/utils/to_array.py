from collections.abc import Sequence

import numpy as np
import pandas as pd


def to_array(record: np.ndarray | pd.Series | Sequence[object]) -> np.ndarray:
    """
    Convert a record to a flat NumPy array of dtype object.

    Args:
        record: numpy.ndarray, pandas.Series, or sequence of feature values.

    Returns:
        np.ndarray: 1D array of feature values, with original dtype preserved.
    """
    if isinstance(record, np.ndarray):
        return record.flatten()
    if isinstance(record, pd.Series):
        return record.to_numpy()
    return np.asarray(record)
