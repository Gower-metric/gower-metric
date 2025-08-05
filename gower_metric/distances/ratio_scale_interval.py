import numpy as np

from gower_metric.utils.missing import apply_missing_strategy, is_missing


def ratio_scale_distance_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    ratio_indices: list[int],
    ranges: np.ndarray,
    h: np.ndarray,
    missing_strategy: str = "ignore",
    weights: np.ndarray | None = None,
    scale_window: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute range-scaled Gower component for ratio-scale features.

    Args:
        X: array (n_x, n_features)
        Y: array (n_y, n_features)
        ratio_indices: indices of ratio-scale columns
        ranges: 1D array of ranges for each ratio-scale column
        h: optional 1D array of bandwidths for KDE scaling
        missing_strategy: one of "ignore", "max_dist", "raise_error"
        weights: optional 1D array of same length as ratio_indices
        scale_window: optional scaling window method

    Returns:
        sum_diff: (n_x, n_y) weighted sum of per-feature distances
        count_present: (n_x, n_y) number of non-missing pairs per feature
    """
    n_x, n_y = X.shape[0], Y.shape[0]
    if not ratio_indices:
        return np.zeros((n_x, n_y), dtype=float), np.zeros((n_x, n_y), dtype=int)

    sum_diff = np.zeros((n_x, n_y), dtype=float)
    count_present = np.zeros((n_x, n_y), dtype=float)

    for pos, j in enumerate(ratio_indices):
        col_x = X[:, j].astype(float)
        col_y = Y[:, j].astype(float)

        mask_x = ~np.array([is_missing(v) for v in col_x])
        mask_y = ~np.array([is_missing(v) for v in col_y])
        present = mask_x[:, None] & mask_y[None, :]

        raw = np.abs(col_x[:, None] - col_y[None, :])
        if ranges[pos] > 0:
            diff = raw / ranges[pos]
            diff[diff > 1.0] = 1.0
        else:
            diff = np.zeros_like(raw)

        if scale_window in ("kde", "kNN") and h.size > 0:
            diff[raw <= h[pos]] = 0.0

        diff, mask = apply_missing_strategy(diff, present, missing_strategy)

        w = weights[pos] if weights is not None else 1.0

        if w != 1.0:
            sum_diff += diff * w
            count_present += mask.astype(float) * w
        else:
            sum_diff += diff
            count_present += mask.astype(float)

    return sum_diff, count_present
