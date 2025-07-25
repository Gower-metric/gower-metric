from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def to_array(record: np.ndarray | pd.Series | Sequence[object]) -> NDArray[np.object_]:
    """
    Convert a record to a flat NumPy array of dtype object.

    Args:
        record: numpy.ndarray, pandas.Series, or sequence of feature values.

    Returns:
        np.ndarray: 1D array of feature values.
    """
    if isinstance(record, np.ndarray):
        return record.flatten()
    if isinstance(record, pd.Series):
        return record.to_numpy()
    return np.array(record, dtype=object)
