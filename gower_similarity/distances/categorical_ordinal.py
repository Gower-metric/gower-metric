import numpy as np
from typing import List, Tuple, Optional
from ..utils.missing import is_missing
from ..utils.cat_ord_ut import get_ranks_mapping, get_cardinalities_mapping


def ordinal_distance_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    ordinal_indices: List[int],
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the ordinal categorical component of Gower distance between rows of X and Y.

    Args:
        X (np.ndarray): First dataset, shape (n_x, n_features).
        Y (np.ndarray): Second dataset, shape (n_y, n_features).
        ordinal_indices (List[int]): Indices of ordinal features.
        weights (Optional[np.ndarray]): Optional weight per ordinal feature.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - sum_diff: matrix (n_x, n_y) of weighted, normalized ordinal distances
            - count_present: matrix (n_x, n_y) of counts of present (non-missing) features
    """
    n_x, n_y = X.shape[0], Y.shape[0]
    sum_diff = np.zeros((n_x, n_y), dtype=float)
    count_present = np.zeros((n_x, n_y), dtype=int)

    if not ordinal_indices:
        return sum_diff, count_present

    for pos, j in enumerate(ordinal_indices):
        col_x = X[:, j]
        col_y = Y[:, j]

        mask_x = np.array([not is_missing(v) for v in col_x], dtype=bool)
        mask_y = np.array([not is_missing(v) for v in col_y], dtype=bool)
        present = mask_x[:, None] & mask_y[None, :]

        combined = np.concatenate([col_x, col_y])
        ranks_map, min_rank, max_rank = get_ranks_mapping(combined)
        _, counts_list = get_cardinalities_mapping(combined)
        counts_arr = np.array(counts_list, dtype=float)

        r_x = np.array(
            [ranks_map[v] if not is_missing(v) else np.nan for v in col_x],
            dtype=float)
        r_y = np.array(
            [ranks_map[v] if not is_missing(v) else np.nan for v in col_y],
            dtype=float)

        diff = np.abs(r_x[:, None] - r_y[None, :])

        mid = (counts_arr - 1) / 2.0
        mid_x = mid[r_x.astype(int)][:, None]
        mid_y = mid[r_y.astype(int)][None, :]

        denom = (max_rank - min_rank - mid[0] - mid[-1])

        dist = (diff - mid_x - mid_y) / denom
        dist[~present] = 0.0

        # TODO: later add support for more advanced weights implementation
        w = float(weights[pos]) if weights is not None else 1.0
        sum_diff += dist * w
        count_present += present.astype(int)
        
        # TODO: add normalization

    return sum_diff, count_present
