from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def to_array(record: np.ndarray | pd.Series | Sequence[object]) -> NDArray[Any]:
    """
    Convert a record to a flat NumPy array of dtype object.

    Args:
        record: numpy.ndarray, pandas.Series, or sequence of feature values.

    Returns:
        np.ndarray: 1D array of feature values, with original dtype preserved.
    """
    if isinstance(record, np.ndarray):
        return np.asarray(record.flatten(), dtype=object)
    if isinstance(record, pd.Series):
        return np.asarray(record.to_numpy(dtype=object))
    return np.asarray(record, dtype=object)
