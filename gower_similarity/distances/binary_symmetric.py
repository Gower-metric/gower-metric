import numpy as np
from typing import List, Tuple, Optional

from ..utils.missing import is_missing, apply_missing_strategy


def binary_symmetric_distance_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    binary_indices: List[int],
    missing_strategy: str = "ignore",
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the symmetric binary component of Gower distance between rows of X and Y.

    - Similarity s_ijt = 1 if x_it == x_jt, else 0.
    - δ_ijt (present) = 1 if both values are non-missing, else 0.
    - Distance d_ijt = 1 - s_ijt for δ_ijt = 1; ignored otherwise.

    Args:
        X (np.ndarray): shape (n_x, n_features).
        Y (np.ndarray): shape (n_y, n_features).
        binary_indices (List[int]): Indices of binary symmetric features.
        missing_strategy (str): Strategy for handling missing values, default is 'ignore'.
        weights (Optional[np.ndarray]): Optional weight per binary feature.

    Returns:
        sum_diff: np.ndarray (n_x, n_y) weighted sum of d_ijt
        count_present: np.ndarray (n_x, n_y) of δ_ijt counts
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
        present = mask_x[:, None] & mask_y[None, :]

        # s_ijt: 1 if pairs of values are equal, else 0 -> (0,0) and (1,1)
        equal = (col_x[:, None] == col_y[None, :])
        diff = (present & ~equal).astype(float)

        diff, mask = apply_missing_strategy(diff, present, missing_strategy)

        w = float(weights[pos]) if weights is not None else 1.0
        sum_diff += diff * w
        count_present += mask.astype(float) * w

    return sum_diff, count_present
