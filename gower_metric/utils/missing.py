import math
import pandas as pd
import numpy as np

from typing import Any, Sequence, Optional, Tuple

def is_missing(value: Any) -> bool:
    """
    Returns True if the value is considered missing.
    
    Args:
        value: The value to check.
        
    Returns:
        bool: True if the value is missing, False otherwise.
    """

    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    if hasattr(pd, "isna") and pd.isna(value):
        return True

    return False


def first_not_missing(sequence: Sequence) -> Optional[Any]:
    """
    Returns the first non-missing value from a sequence.
    
    Args:
        sequence: A sequence of values.
        
    Returns:
        Optional[Any]: The first non-missing value, or None if all values are missing.
    """

    for value in sequence:
        if not is_missing(value):
            return value
    return None

def apply_missing_strategy(
    diff: np.ndarray,
    present: np.ndarray,
    nan_method: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the chosen missing-values strategy to the raw diff matrix.

    Args:
        diff: raw distance matrix for one feature, shape (n_x, n_y).
        present: boolean mask where True means both values were non-missing.
        nan_method: one of "ignore", "max_dist", "raise_error".

    Returns:
        diff: adjusted distance matrix
        count_mask: int matrix of same shape, how much to add to count_present
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