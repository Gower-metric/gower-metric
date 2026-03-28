import warnings

import numpy as np


def scale_method(valid: np.ndarray, method: str) -> float:
    """Compute the scaling span for a 1D array of valid (non-NaN) values.

    Args:
        valid (np.ndarray): 1D array of floats (no NaNs).
        method (str): 'range' or 'iqr'.

    Returns:
        float:
            - span: max-min for 'range', Q3-Q1 for 'iqr', or 0.0 if constant/empty.

    Raises:
        ValueError: if method is unknown.

    """
    if valid.size == 0:
        return 0.0

    if method == "range":
        span = valid.max() - valid.min()
    elif method == "iqr":
        q75, q25 = np.percentile(valid, [75, 25])
        span = q75 - q25
    else:
        msg = f"Unknown method '{method}' for scaling."
        raise ValueError(msg)

    return span if span > 0 else 0.0


def get_numeric_ranges(
    X: np.ndarray,
    indices: list[int],
    method: str = "range",
) -> np.ndarray:
    """Compute the range for each numeric column in X based on selected scale method.

    Applied only to ratio-scale and interval-scale data types.

    Args:
        X (np.ndarray): array of shape (n_samples, n_features).
        indices (list[int]): list of column indices to treat as numeric.
        method (str): method for scaling, either 'range' or 'iqr'.

    Returns:
        np.ndarray:
            - 1D array of length len(indices), where each entry is max(X[:, idx]) - min(X[:, idx]). For now we ignore NaNs. If all values are NaN or constant, range is set to 0.0.

    """
    ranges = np.empty(len(indices), dtype=float)
    for pos, j in enumerate(indices):
        col = X[:, j].astype(float)
        valid = col[~np.isnan(col)]
        ranges[pos] = scale_method(valid, method)
    return ranges


def get_numeric_bounds(
    X: np.ndarray,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column min and max for numeric columns, ignoring NaNs.

    Args:
        X (np.ndarray): array of shape (n_samples, n_features).
        indices (list[int]): list of column indices.

    Returns:
        tuple[np.ndarray, np.ndarray]: (mins, maxs), each 1D of length len(indices).

    """
    mins = np.empty(len(indices), dtype=float)
    maxs = np.empty(len(indices), dtype=float)
    for pos, j in enumerate(indices):
        col = X[:, j].astype(float)
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            mins[pos] = np.nan
            maxs[pos] = np.nan
        else:
            mins[pos] = valid.min()
            maxs[pos] = valid.max()
    return mins, maxs


def check_out_of_range(
    X: np.ndarray,
    indices: list[int],
    mins: np.ndarray,
    maxs: np.ndarray,
    feature_label: str,
) -> list[str]:
    """Check which columns have values outside the fitted [min, max].

    Args:
        X (np.ndarray): array of shape (n_samples, n_features).
        indices (list[int]): column indices to check.
        mins (np.ndarray): fitted minimums, length len(indices).
        maxs (np.ndarray): fitted maximums, length len(indices).
        feature_label (str): label like "numeric" or "ratio_scale".

    Returns:
        list[str]: detail strings for offending columns (empty if all in range).

    """
    details: list[str] = []
    for pos, j in enumerate(indices):
        if np.isnan(mins[pos]):
            continue
        col = X[:, j].astype(float)
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            continue
        col_min, col_max = valid.min(), valid.max()
        if col_min < mins[pos] or col_max > maxs[pos]:
            details.append(
                f"{feature_label} column {j}: "
                f"values in [{col_min}, {col_max}], "
                f"fitted range [{mins[pos]}, {maxs[pos]}]",
            )
    return details


def enforce_oor_policy(
    *arrays: np.ndarray,
    strategy: str,
    numeric_indices: list[int],
    numeric_mins: np.ndarray,
    numeric_maxs: np.ndarray,
    ratio_scale_indices: list[int],
    ratio_mins: np.ndarray,
    ratio_maxs: np.ndarray,
    stacklevel: int = 3,
) -> None:
    """Enforce the out-of-range policy for numeric and ratio-scale columns.

    Accepts one or more arrays (like scikit-learn's X, y pattern) and checks
    all of them for values outside the fitted [min, max]. Collects violations
    across all arrays into a single warning or error.
    Does nothing when strategy is 'clip'.

    Args:
        *arrays (np.ndarray): one or more object arrays of shape (n_samples, n_features).
        strategy (str): 'clip', 'warning', or 'error'.
        numeric_indices (list[int]): fitted numeric column indices.
        numeric_mins (np.ndarray): fitted minimums for numeric columns.
        numeric_maxs (np.ndarray): fitted maximums for numeric columns.
        ratio_scale_indices (list[int]): fitted ratio-scale column indices.
        ratio_mins (np.ndarray): fitted minimums for ratio-scale columns.
        ratio_maxs (np.ndarray): fitted maximums for ratio-scale columns.
        stacklevel (int): stack depth for warnings.warn.

    Raises:
        ValueError: when strategy is 'error' and out-of-range values are found.

    """
    if strategy == "clip":
        return

    oor_details: list[str] = []
    for arr in arrays:
        if numeric_indices and numeric_mins.size > 0:
            oor_details.extend(
                check_out_of_range(
                    arr,
                    numeric_indices,
                    numeric_mins,
                    numeric_maxs,
                    "numeric",
                ),
            )
        if ratio_scale_indices and ratio_mins.size > 0:
            oor_details.extend(
                check_out_of_range(
                    arr,
                    ratio_scale_indices,
                    ratio_mins,
                    ratio_maxs,
                    "ratio_scale",
                ),
            )
    if not oor_details:
        return

    msg = (
        "Out-of-range values detected in transform data "
        "(distances will be clipped to [0, 1]):\n" + "\n".join(oor_details)
    )
    if strategy == "error":
        raise ValueError(msg)
    warnings.warn(msg, UserWarning, stacklevel=stacklevel)
