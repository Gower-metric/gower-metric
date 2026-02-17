from typing import Any

import numpy as np
import pandas as pd

MAX_BINARY_UNIQUE_VALUES = 2


def fit_binary_features(
    arr: np.ndarray,
    binary_indices: list[int],
) -> dict[int, dict[str, Any]]:
    """Fit binary features metadata.

    Args:
        arr (np.ndarray): Input data array.
        binary_indices (list[int]): Indices of binary features.

    Returns:
        dict[int, dict[str, Any]]: Metadata for binary features, including mapping and unique values.

    Raises:
        ValueError: If a binary column has more than 2 unique values.

    """
    binary_metadata: dict[int, dict[str, Any]] = {}

    for j in binary_indices:
        col = arr[:, j]
        non_null_mask = ~pd.isna(col)
        valid_values = col[non_null_mask]

        unique_vals = np.unique(valid_values)

        if len(unique_vals) > MAX_BINARY_UNIQUE_VALUES:
            msg = (
                f"Binary feature at index {j} has more than {MAX_BINARY_UNIQUE_VALUES} "
                f"unique values: {unique_vals}. "
                f"Gower metric expects at most {MAX_BINARY_UNIQUE_VALUES} "
                "unique values for binary features."
            )
            raise ValueError(msg)

        unique_vals.sort()

        mapping = {}
        if len(unique_vals) > 0:
            mapping[unique_vals[0]] = 0.0
        if len(unique_vals) > 1:
            mapping[unique_vals[1]] = 1.0

        binary_metadata[j] = {
            "mapping": mapping,
            "values": unique_vals,
        }

    return binary_metadata
