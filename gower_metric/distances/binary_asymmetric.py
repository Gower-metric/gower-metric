from typing import Any

import numpy as np
import pandas as pd

from gower_metric.utils.missing import apply_missing_strategy


def binary_asymmetric_component(
    X: np.ndarray,
    Y: np.ndarray,
    binary_indices: list[int],
    missing_strategy: str = "ignore",
    weights: np.ndarray | None = None,
    metadata: dict[int, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the asymmetric binary component of Gower metric between rows of X and Y.

    Description:
        - Similarity s_ijt = 1 if x_it = x_jt = positive_value, else 0.
        - δ_ijt (present) = 1 if both non-missing and at least one equals positive_value, else 0.
        - Distance d_ijt = 1 - s_ijt for δ_ijt = 1, ignored otherwise.

    Per Gower (1971), for asymmetric binary variables, joint absences (both negative)
    are excluded from the comparison (δ_ijk = 0).

    Args:
        X (np.ndarray): shape (n_x, n_features).
        Y (np.ndarray): shape (n_y, n_features).
        binary_indices (list[int]): indices of asymmetric binary features.
        missing_strategy (str): strategy for handling missing values, default is 'ignore'.
        weights (Optional[np.ndarray]): optional per-feature weights.
        metadata (Optional[dict[int, dict[str, Any]]]): fitted binary metadata per feature index,
            containing 'positive_value' key identifying the presence attribute.
            If None, defaults to comparing with integer 1.

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

        mask_x = ~pd.isna(col_x)
        mask_y = ~pd.isna(col_y)
        valid = mask_x[:, None] & mask_y[None, :]

        positive_val = (
            metadata[j]["positive_value"] if metadata and j in metadata else 1
        )
        present = valid & (
            (col_x[:, None] == positive_val) | (col_y[None, :] == positive_val)
        )
        both_positive = (col_x[:, None] == positive_val) & (
            col_y[None, :] == positive_val
        )
        raw = (~both_positive).astype(float)

        diff, mask = apply_missing_strategy(raw, present, missing_strategy)

        w = weights[pos] if weights is not None else 1.0
        sum_diff += diff * w
        count_present += mask * w

    return sum_diff, count_present
