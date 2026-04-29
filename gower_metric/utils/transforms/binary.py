import warnings
from typing import Any

import numpy as np
import pandas as pd

MAX_BINARY_UNIQUE_VALUES = 2


def _transform_binary(
    col: np.ndarray,
    col_idx: int,
    metadata: dict[str, Any],
    handle_unseen: str,
    binary_type: str,
) -> np.ndarray:
    """Transform a binary feature column into its numeric representation.

    Maps each value to 0.0 or 1.0 based on the fitted mapping, and handles
    missing values (``NaN``) and unseen values according to ``handle_unseen``.

    Args:
        col (np.ndarray): 1-D array of raw values for a single column.
        col_idx (int): Column index, used in error and warning messages.
        metadata (dict[str, Any]): Fitted metadata produced by
            :func:`~gower_metric.utils.binary_ut.fit_binary_features`, containing
            ``"mapping"``, ``"values"``, and ``"is_explicit_order"`` keys.
        handle_unseen (str): Strategy for unseen values — ``"error"``,
            ``"warning"``, or ``"missing"``.
        binary_type (str): The binary feature type name (e.g. ``"binary_asymmetric"``).

    Returns:
        np.ndarray: 1-D float array of the same length as *col*, with values
        mapped to 0.0, 1.0, or ``np.nan``.

    Raises:
        ValueError: If an unseen value is encountered and *handle_unseen* is
            ``"error"``, or if an explicit value order is violated.

    """
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
            short_type = binary_type.replace("binary_", "")
            msg = (
                f"Binary {short_type} column {col_idx} has {total_unique} unique values total "
                f"(fitted: {sorted(fitted_vals)}, unseen: {sorted(unseen_vals)}). "
                f"Binary features must have at most {MAX_BINARY_UNIQUE_VALUES} values. "
                f"Consider using {binary_type}_value_order to explicitly define the expected binary values, "
                "or change the feature type if this is not actually a binary feature."
            )
            raise ValueError(msg)

    for i, v in enumerate(col):
        if pd.isna(v):
            transformed_col[i] = np.nan
        elif v not in mapping:
            if is_explicit:
                unseen_msg = (
                    f"Value '{v}' in column {col_idx} violates {binary_type}_value_order. "
                    f"Expected one of {list(mapping.keys())}."
                )
                raise ValueError(unseen_msg)
            unseen_msg = (
                f"Value '{v}' in column {col_idx} not found in fitted binary mapping. "
                f"Expected one of {list(mapping.keys())}."
            )
            if handle_unseen == "error":
                raise ValueError(unseen_msg)
            if handle_unseen == "warning":
                warnings.warn(
                    f"{unseen_msg} Treating as missing (np.nan).",
                    UserWarning,
                    stacklevel=3,
                )
            transformed_col[i] = np.nan
        else:
            transformed_col[i] = mapping[v]

    return transformed_col


def transform_binary_asymmetric(
    col: np.ndarray,
    col_idx: int,
    metadata: dict[str, Any],
    handle_unseen: str,
) -> np.ndarray:
    """Transform a binary asymmetric feature column into its numeric representation."""
    return _transform_binary(col, col_idx, metadata, handle_unseen, "binary_asymmetric")


def transform_binary_symmetric(
    col: np.ndarray,
    col_idx: int,
    metadata: dict[str, Any],
    handle_unseen: str,
) -> np.ndarray:
    """Transform a binary symmetric feature column into its numeric representation."""
    return _transform_binary(col, col_idx, metadata, handle_unseen, "binary_symmetric")
