import numpy as np

from gower_metric.utils.missing import apply_missing_strategy, is_missing


def numeric_component(
    X: np.ndarray,
    Y: np.ndarray,
    numeric_indices: list[int],
    ranges: np.ndarray,
    h: np.ndarray,
    missing_strategy: str = "ignore",
    weights: np.ndarray | None = None,
    scale_window: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute range-scaled Gower component for interval-scale (numeric) features. The same logic as
    in ratio scale.

    Args:
        X (np.ndarray): array of shape (n_x, n_features)
        Y (np.ndarray): array of shape (n_y, n_features)
        numeric_indices (list[int]): indices of numeric-interval columns
        ranges (np.ndarray): 1D array of ranges for each numeric-interval column
        h (np.ndarray): optional 1D array of bandwidths for KDE scaling
        missing_strategy (str): one of "ignore", "max_dist", "raise_error"
        weights (Optional[np.ndarray]): optional 1D array of same length as numeric_indices
        scale_window (Optional[str]): optional scaling window method

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - sum_diff: matrix (n_x, n_y) weighted sum of per-feature distances
            - count_present: matrix (n_x, n_y) number of non-missing pairs per feature
    """
    n_x, n_y = X.shape[0], Y.shape[0]
    if not numeric_indices:
        return np.zeros((n_x, n_y), float), np.zeros((n_x, n_y), float)

    sum_diff = np.zeros((n_x, n_y), float)
    count_present = np.zeros((n_x, n_y), float)

    for pos, j in enumerate(numeric_indices):
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
        sum_diff += diff * w
        count_present += mask.astype(float) * w

    return sum_diff, count_present
