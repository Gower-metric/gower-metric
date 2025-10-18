import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def is_missing(value: Any) -> bool:
    """
    Return True if the value is considered missing.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is missing, False otherwise.
    """
    return bool(
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or (hasattr(pd, "isna") and pd.isna(value))
    )


def first_not_missing(sequence: Sequence) -> Any | None:
    """
    Return the first non-missing value from a sequence.

    Args:
        sequence (Sequence): A sequence of values.

    Returns:
        Optional[Any]: The first non-missing value, or None if all values are missing.
    """
    for value in sequence:
        if not is_missing(value):
            return value
    return None


def apply_missing_strategy(
    diff: np.ndarray, present: np.ndarray, nan_method: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the chosen missing-values strategy to the raw diff matrix.

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
        count_mask = present.astype(int)

    elif nan_method == "raise_error":
        if not present.all():
            raise ValueError
        count_mask = present.astype(int)

    else:
        raise ValueError(f"Unknown nan_method '{nan_method}'")

    return diff, count_mask
