import numpy as np
from typing import List, Tuple, Optional

from ..utils.missing import is_missing


def ratio_scale_distance_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    ratio_indices: List[int],
    ranges: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic range-scaled Gower component for ratio-scale features.

    Args:
        X: array (n_x, n_features)
        Y: array (n_y, n_features)
        ratio_indices: indices of ratio-scale columns
        ranges: 1D array of ranges for each ratio-scale column
        weights: optional 1D array of same length as ratio_indices

    Returns:
        sum_diff: (n_x, n_y) weighted sum of per-feature distances
        count_present: (n_x, n_y) number of non-missing pairs per feature
    """
    n_x, n_y = X.shape[0], Y.shape[0]
    if not ratio_indices:
        return np.zeros((n_x, n_y), dtype=float), np.zeros((n_x, n_y),
                                                           dtype=int)

    sum_diff = np.zeros((n_x, n_y), dtype=float)
    count_present = np.zeros((n_x, n_y), dtype=int)

    for pos, j in enumerate(ratio_indices):
        col_x = X[:, j].astype(float)
        col_y = Y[:, j].astype(float)

        mask_x = ~np.array([is_missing(v) for v in col_x])
        mask_y = ~np.array([is_missing(v) for v in col_y])
        present = mask_x[:, None] & mask_y[None, :]

        # TODO: add IQR, KDE. Maybe create a separate function/file for this??
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
