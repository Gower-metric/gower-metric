from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def first_not_missing(sequence: Sequence) -> Any | None:
    """Return the first non-missing value from a sequence.

    Args:
        sequence (Sequence): A sequence of values.

    Returns:
        Optional[Any]: The first non-missing value, or None if all values are missing.

    """
    for value in sequence:
        if not pd.isna(value):
            return value
    return None


def apply_missing_strategy(
    diff: np.ndarray,
    present: np.ndarray,
    nan_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the chosen missing-values strategy to the raw diff matrix.

    Args:
        diff (np.ndarray): raw distance matrix for one feature, shape (n_x, n_y).
        present (np.ndarray): boolean mask where True means both values were non-missing.
        nan_method (str): one of "ignore", "max_dist", "raise_error".

    Returns:
        tuple[np.ndarray, np.ndarray]: Diff is adjusted distance matrix. Count_mask is int matrix of same shape, how much to add to count_present.

    Raises:
        ValueError: if nan_method is not recognized.

    """
    if nan_method == "ignore":
        diff[~present] = 0.0
        count_mask = present.astype(int)

    elif nan_method == "max_dist":
        diff[~present] = 1.0
        count_mask = np.ones_like(present, dtype=int)

    elif nan_method == "raise_error":
        if not present.all():
            msg = (
                "Missing values detected in data. "
                "Set missing_strategy='ignore' or 'max_dist' to handle them."
            )
            raise ValueError(msg)
        count_mask = present.astype(int)

    else:
        msg = f"Unknown nan_method '{nan_method}'"
        raise ValueError(msg)

    return diff, count_mask
