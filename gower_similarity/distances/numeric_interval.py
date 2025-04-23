import numpy as np

from typing import Tuple, List

def get_numeric_ranges(data: np.ndarray, numeric_indices: List[int]) -> np.ndarray:
    """
    Compute the ranges (max - min) for each numeric feature, ignoring missing values.

    Args:
        data (np.ndarray): The input data array.
        numeric_indices (List[int]): List of indices for numeric features.

    Returns:
        np.ndarray: An array containing the ranges for each numeric feature.
    """
    if data.size == 0 or not numeric_indices:
        return np.array([], dtype=float)
    
    cols = data[:, numeric_indices].astype(np.float64)
    min_vals = np.nanmin(cols, axis=0)
    max_vals = np.nanmax(cols, axis=0)

    ranges = max_vals - min_vals

    # TODO: add feature to let user choose how to handle NaN values
    ranges = np.where(np.isnan(ranges) | np.isclose(ranges, 0.0), 0.0, ranges)
    return ranges

def numeric_distance_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    numeric_indices: List[int],
    ranges: np.ndarray,
    weights: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute numeric component of Gower distance between rows of X and Y.
    
    Args:
        X (np.ndarray): First input data array.
        Y (np.ndarray): Second input data array.
        numeric_indices (List[int]): List of indices for numeric features.
        ranges (np.ndarray): Ranges for each numeric feature.
        weights (np.ndarray): Weights for each feature.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the distance matrix and the range matrix.
    """
    n_x, n_y = X.shape[0], Y.shape[0]
    if not numeric_indices:
        return np.zeros((n_x, n_y)), np.zeros((n_x, n_y))
    
    Xnumeric = X[:, numeric_indices].astype(np.float64)
    Ynumeric = Y[:, numeric_indices].astype(np.float64)

    # mask
    mask_x = ~np.isnan(Xnumeric)
    mask_y = ~np.isnan(Ynumeric)

    present = mask_x[:, None, :] & mask_y[None, :, :]

    # abs difference
    diff = np.abs(Xnumeric[:, None, :] - Ynumeric[None, :, :])

    # normalize
    ranges_safe = ranges.copy()
    ranges_safe[ranges_safe == 0.0] = 1.0
    diff_norm = diff / ranges_safe
    diff_norm[..., ranges == 0.0] = 0.0
    diff_norm[~present] = 0.0

    # apply weights
    if weights is not None:
        w = weights.reshape(1, 1, -1)
        diff_weighted = diff_norm * w
    else:
        diff_weighted = diff_norm

    summ_diff = np.sum(diff_weighted, axis=2)
    count_present = np.sum(present, axis=2)

    return summ_diff, count_present
    
