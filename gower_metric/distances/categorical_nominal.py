import numpy as np

from gower_metric.utils.missing import apply_missing_strategy, is_missing


def categorical_nominal_component(
    X: np.ndarray,
    Y: np.ndarray,
    categorical_indices: list[int],
    missing_strategy: str = "ignore",
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the nominal categorical component of Gower metric between rows of X and Y.

    Args:
        X (np.ndarray): First dataset, shape (n_x, n_features).
        Y (np.ndarray): Second dataset, shape (n_y, n_features).
        categorical_indices (list[int]): Indices of nominal categorical features.
        missing_strategy (str): Strategy for handling missing values, default is "ignore".
        weights (Optional[np.ndarray]): Optional weight per categorical feature.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - sum_diff: matrix (n_x, n_y) of weighted counts of differing features
            - count_present: matrix (n_x, n_y) of counts of present (non-missing) features

    """
    n_x, n_y = X.shape[0], Y.shape[0]
    if not categorical_indices:
        return np.zeros((n_x, n_y), dtype=float), np.zeros((n_x, n_y), dtype=float)

    sum_diff = np.zeros((n_x, n_y), dtype=float)
    count_present = np.zeros((n_x, n_y), dtype=float)

    for pos, j in enumerate(categorical_indices):
        col_x = X[:, j]
        col_y = Y[:, j]

        mask_x = np.array([not is_missing(v) for v in col_x], dtype=bool)
        mask_y = np.array([not is_missing(v) for v in col_y], dtype=bool)
        present = mask_x[:, None] & mask_y[None, :]

        diff = (~(col_x[:, None] == col_y[None, :]) & present).astype(float)
        diff, mask = apply_missing_strategy(diff, present, missing_strategy)

        w = weights[pos] if weights is not None else 1.0
        sum_diff += diff * w
        count_present += mask.astype(float) * w

    return sum_diff, count_present
