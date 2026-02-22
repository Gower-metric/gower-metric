import warnings
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
                    f"Binary feature at index {j} contains values not in binary_asymmetric_value_order: "
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


def transform_binary_asymmetric(
    col: np.ndarray,
    col_idx: int,
    metadata: dict[str, Any],
    handle_unseen: str,
) -> np.ndarray:
    """Transform a binary asymmetric feature column."""
    transformed_col = np.zeros(col.shape[0], dtype=float)
    mapping = metadata["mapping"]
    is_explicit = metadata.get("is_explicit_order", False)

    if not is_explicit:
        non_null_mask = ~pd.isna(col)
        unique_transform_vals = set(np.unique(col[non_null_mask]))
        fitted_vals = set(mapping.keys())
        unseen_vals = unique_transform_vals - fitted_vals
        total_unique = len(fitted_vals) + len(unseen_vals)

        if total_unique > MAX_BINARY_UNIQUE_VALUES:
            msg = (
                f"Binary asymmetric column {col_idx} has {total_unique} unique values total "
                f"(fitted: {sorted(fitted_vals)}, unseen: {sorted(unseen_vals)}). "
                f"Binary features must have at most {MAX_BINARY_UNIQUE_VALUES} values. "
                "Consider using binary_asymmetric_value_order to explicitly define the expected binary values, "
                "or change the feature type if this is not actually a binary feature."
            )
            raise ValueError(msg)

    for i, v in enumerate(col):
        if pd.isna(v):
            transformed_col[i] = np.nan
        elif v not in mapping:
            if is_explicit:
                msg = (
                    f"Value '{v}' in column {col_idx} violates binary_asymmetric_value_order. "
                    f"Expected only {list(mapping.keys())}, but got '{v}'. "
                    "This suggests data quality issues or incorrect value order specification."
                )
                raise ValueError(msg)
            if handle_unseen == "error":
                msg = (
                    f"Value '{v}' in column {col_idx} not found in fitted binary mapping "
                    f"{list(mapping.keys())}. Set handle_unseen_binary_asymmetric='missing' or "
                    "'warning' in Config to handle unseen values gracefully."
                )
                raise ValueError(msg)
            if handle_unseen == "warning":
                warnings.warn(
                    f"Value '{v}' in column {col_idx} not found in fitted binary mapping "
                    f"{list(mapping.keys())}. Treating as missing (np.nan).",
                    UserWarning,
                    stacklevel=2,
                )
                transformed_col[i] = np.nan
            else:
                transformed_col[i] = np.nan
        else:
            transformed_col[i] = mapping[v]

    return transformed_col


def transform_binary_symmetric(
    col: np.ndarray,
    col_idx: int,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Transform a binary symmetric feature column."""
    transformed_col = np.zeros(col.shape[0], dtype=float)
    mapping = metadata["mapping"]

    for i, v in enumerate(col):
        if pd.isna(v):
            transformed_col[i] = np.nan
        else:
            if v not in mapping:
                if len(mapping) < MAX_BINARY_UNIQUE_VALUES:
                    transformed_col[i] = np.nan
                    continue

                msg = (
                    f"Value '{v}' in column {col_idx} not found in fitted binary mapping "
                    f"{list(mapping.keys())}."
                )
                raise ValueError(msg)
            transformed_col[i] = mapping[v]

    return transformed_col
