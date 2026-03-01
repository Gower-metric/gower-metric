from typing import Any

import numpy as np
import pandas as pd

MAX_BINARY_UNIQUE_VALUES = 2


def fit_binary_features(
    arr: np.ndarray,
    binary_indices: list[int],
    binary_value_order: dict[int, list[Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Fit binary features metadata.

    Args:
        arr (np.ndarray): Input data array.
        binary_indices (list[int]): Indices of binary features.
        binary_value_order (dict[int, list[Any]] | None): Optional explicit ordering of binary values.
            If provided, specifies expected values for each binary column (must be exactly 2 values).
            Creates full mapping for all expected values, even if not present in training data.
            If None, values are auto-detected from training data.

    Returns:
        dict[int, dict[str, Any]]: Metadata for binary features, including:
            - mapping: dict mapping values to 0.0/1.0
            - values: array of unique values (or expected values if explicit order provided)
            - is_explicit_order: bool indicating if explicit order was used

    Raises:
        ValueError: If a binary column has more than 2 unique values, or if training data
            contains values not in binary_value_order when explicit ordering is used.

    """
    binary_metadata: dict[int, dict[str, Any]] = {}

    for j in binary_indices:
        col = arr[:, j]
        non_null_mask = ~pd.isna(col)
        valid_values = col[non_null_mask]

        if binary_value_order is not None and j in binary_value_order:
            expected_vals = binary_value_order[j]
            unique_vals = np.array(expected_vals)

            actual_unique = np.unique(valid_values)
            unexpected = set(actual_unique) - set(expected_vals)
            if unexpected:
                msg = (
                    f"Binary feature at index {j} contains unexpected values: "
                    f"{unexpected}. Expected values: {expected_vals}"
                )
                raise ValueError(msg)

            mapping = {expected_vals[0]: 0.0, expected_vals[1]: 1.0}

            binary_metadata[j] = {
                "mapping": mapping,
                "values": unique_vals,
                "is_explicit_order": True,
            }

        else:
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
                "is_explicit_order": False,
            }

    return binary_metadata
