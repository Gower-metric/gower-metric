import numpy as np

from typing import List


def scale_method(valid: np.ndarray, method: str) -> float:
    """
    Compute the scaling span for a 1D array of valid (non-NaN) values.

    Args:
        valid: 1D array of floats (no NaNs).
        method: 'range' or 'iqr'.

    Returns:
        span (float): max-min for 'range', Q3-Q1 for 'iqr', or 0.0 if constant/empty.
    """
    if valid.size == 0:
        return 0.0

    if method == 'range':
        span = valid.max() - valid.min()
    elif method == "iqr":
        q75, q25 = np.percentile(valid, [75, 25])
        span = q75 - q25
    else:
        raise ValueError(f"Unknown method '{method}' for scaling.")

    return span if span > 0 else 0.0


def get_numeric_ranges(
    X: np.ndarray,
    indices: List[int],
    method: str = 'range',
) -> np.ndarray:
    """
    Compute the range for each numeric column in X based on selected scale method. Applied only to
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
        ranges[pos] = scale_method(valid, method)
    return ranges
