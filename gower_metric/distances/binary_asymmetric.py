import numpy as np

from gower_metric.utils.missing import apply_missing_strategy, is_missing


def binary_asymmetric_component(
    X: np.ndarray,
    Y: np.ndarray,
    binary_indices: list[int],
    missing_strategy: str = "ignore",
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the asymmetric binary component of Gower metric between rows of X and Y.

    Description:
        - Similarity s_ijt = 1 if x_it = x_jt = 1, else 0.
        - δ_ijt (present) = 1 if both non-missing and at least one equals 1, else 0.
        - Distance d_ijt = 1 - s_ijt for δ_ijt = 1, ignored otherwise.

    Args:
        X (np.ndarray): shape (n_x, n_features).
        Y (np.ndarray): shape (n_y, n_features).
        binary_indices (list[int]): indices of asymmetric binary features.
        missing_strategy (str): strategy for handling missing values, default is 'ignore'.
        weights (Optional[np.ndarray]): optional per-feature weights.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - sum_diff: matrix (n_x, n_y); weighted sum of d_ijt
            - count_present: matrix (n_x, n_y); δ_ijt counts
    """
    n_x, n_y = X.shape[0], Y.shape[0]
    sum_diff = np.zeros((n_x, n_y), dtype=float)
    count_present = np.zeros((n_x, n_y), dtype=float)

    if not binary_indices:
        return sum_diff, count_present

    for pos, j in enumerate(binary_indices):
        col_x = X[:, j]
        col_y = Y[:, j]

        mask_x = np.array([not is_missing(v) for v in col_x], dtype=bool)
        mask_y = np.array([not is_missing(v) for v in col_y], dtype=bool)
        valid = mask_x[:, None] & mask_y[None, :]

        # δ_ijt: at least one presence (1) and both non-missing
        present = valid & ((col_x[:, None] == 1) | (col_y[None, :] == 1))

        # s_ijt: 1 only if both == 1
        both_one = (col_x[:, None] == 1) & (col_y[None, :] == 1)
        raw = (~both_one).astype(float)

        diff, mask = apply_missing_strategy(raw, present, missing_strategy)

        w = weights[pos] if weights is not None else 1.0
        sum_diff += diff * w
        count_present += mask.astype(float) * w

    return sum_diff, count_present
