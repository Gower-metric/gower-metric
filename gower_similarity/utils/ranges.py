import numpy as np
from typing import List


def get_numeric_ranges(
    X: np.ndarray,
    indices: List[int],
) -> np.ndarray:
    """
    Compute the range (max - min) for each numeric column in X. Applied only to
    ratio-scale and internal-scale data types.

    Args:
        X: array of shape (n_samples, n_features), dtype = float or object convertible to float
        indices: list of column indices to treat as numeric

    Returns:
        1D array of length len(indices), where each entry is
        max(X[:, idx]) - min(X[:, idx]), for now we ignore NaNs.
        If all values are NaN or constant, range is set to 0.0.
    """
    # TODO: add feature to let user choose how to handle NaN values
    ranges = np.empty(len(indices), dtype=float)
    for pos, j in enumerate(indices):
        col = X[:, j].astype(float)
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            ranges[pos] = 0.0
        else:
            ranges[pos] = valid.max() - valid.min()
    return ranges
