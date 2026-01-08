from typing import Any

import numpy as np

from gower_metric.utils.missing import apply_missing_strategy, is_missing


def categorical_ordinal_component(
    X: np.ndarray,
    Y: np.ndarray,
    ordinal_indices: list[int],
    metadata: dict[int | str, dict[str, Any]],
    missing_strategy: str = "ignore",
    calculation_type: str = "kaufman",
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the ordinal categorical component of Gower metric between rows of X and Y.

    Args:
        X (np.ndarray): First dataset, shape (n_x, n_features).
        Y (np.ndarray): Second dataset, shape (n_y, n_features).
        ordinal_indices (list[int]): Indices of ordinal features.
        metadata (dict[int, dict[str, Any]]): Metadata for ordinal features. Computed
            by Gower.fit() on whole data range and passed to the distance function.
        missing_strategy (str): Strategy for handling missing values, default is "ignore".
        calculation_type (str): Type of calculation for ordinal distance, available options are
            "kaufman" and "podani". Default is "kaufman".
        weights (Optional[np.ndarray]): Optional weight per ordinal feature.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - sum_diff: matrix (n_x, n_y) of weighted, normalized ordinal distances
            - count_present: matrix (n_x, n_y) of counts of present (non-missing) features

    """
    n_x, n_y = X.shape[0], Y.shape[0]
    sum_diff = np.zeros((n_x, n_y), dtype=float)
    count_present = np.zeros((n_x, n_y), dtype=float)

    if not ordinal_indices:
        return sum_diff, count_present

    for pos, j in enumerate(ordinal_indices):
        col_x = X[:, j]
        col_y = Y[:, j]

        mask_x = np.array([not is_missing(v) for v in col_x], dtype=bool)
        mask_y = np.array([not is_missing(v) for v in col_y], dtype=bool)
        present = mask_x[:, None] & mask_y[None, :]

        if metadata and j in metadata:
            info = metadata[j]
            ranks_map = info["ranks"]
            min_rank = info["min"]
            max_rank = info["max"]
            counts_arr = info["counts"]
        else:
            msg = f"Missing metadata for ordinal feature at index {j}."
            raise ValueError(msg)

        if min_rank is None:
            continue

        r_x = np.array([ranks_map.get(v, np.nan) for v in col_x], dtype=float)
        r_y = np.array([ranks_map.get(v, np.nan) for v in col_y], dtype=float)

        if calculation_type == "kaufman":
            denom = max_rank - min_rank

            dist = (
                np.zeros((n_x, n_y), dtype=float)
                if denom == 0
                else np.abs(r_x[:, None] - r_y[None, :]) / denom
            )

        else:
            diff = np.abs(r_x[:, None] - r_y[None, :])
            mid = (counts_arr - 1) / 2.0
            mid_x = mid[r_x.astype(int)][:, None]
            mid_y = mid[r_y.astype(int)][None, :]
            podani_denom = max_rank - min_rank - mid[0] - mid[-1]

            if podani_denom <= 0:
                # fallback to kaufman if podani denominator is not valid
                base_denom = max_rank - min_rank
                dist = (
                    np.zeros((n_x, n_y), dtype=float)
                    if base_denom == 0
                    else diff / base_denom
                )
            else:
                dist = (diff - mid_x - mid_y) / podani_denom
                dist = np.clip(dist, 0.0, 1.0)

        dist, mask = apply_missing_strategy(dist, present, missing_strategy)

        w = weights[pos] if weights is not None else 1.0
        sum_diff += dist * w
        count_present += mask.astype(float) * w

    return sum_diff, count_present
