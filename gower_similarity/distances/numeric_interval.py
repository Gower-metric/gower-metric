import numpy as np
from typing import List, Tuple, Optional

from ..utils.ranges import get_numeric_ranges
from ..utils.missing import is_missing


def numeric_distance_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    numeric_indices: List[int],
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic range-scaled Gower component for interval-scale (numeric) features. The same logic as
    in ratio scale.

    Args:
        X: array of shape (n_x, n_features)
        Y: array of shape (n_y, n_features)
        numeric_indices: indices of numeric-interval columns
        weights: optional 1D array of same length as numeric_indices

    Returns:
        sum_diff: (n_x, n_y) weighted sum of per-feature distances
        count_present: (n_x, n_y) number of non-missing pairs per feature
    """
    n_x, n_y = X.shape[0], Y.shape[0]
    if not numeric_indices:
        return np.zeros((n_x, n_y), float), np.zeros((n_x, n_y), int)

    ranges = get_numeric_ranges(np.vstack([X, Y]), numeric_indices)

    sum_diff = np.zeros((n_x, n_y), float)
    count_present = np.zeros((n_x, n_y), int)

    for pos, j in enumerate(numeric_indices):
        col_x = X[:, j].astype(float)
        col_y = Y[:, j].astype(float)

        mask_x = ~np.array([is_missing(v) for v in col_x])
        mask_y = ~np.array([is_missing(v) for v in col_y])
        present = mask_x[:, None] & mask_y[None, :]

        raw = np.abs(col_x[:, None] - col_y[None, :])

        if ranges[pos] > 0:
            diff = raw / ranges[pos]
        else:
            diff = np.zeros_like(raw)

        diff[~present] = 0.0

        w = weights[pos] if weights is not None else 1.0
        sum_diff += diff * w
        count_present += present.astype(int)

    return sum_diff, count_present
