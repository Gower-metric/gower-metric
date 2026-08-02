# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

import warnings

import numpy as np
import pandas as pd

from gower_metric._typing import AnyArray, BinaryMetadata, FloatArray

MAX_BINARY_UNIQUE_VALUES = 2


def _transform_binary(
    col: AnyArray,
    col_idx: int,
    metadata: BinaryMetadata,
    handle_unseen: str,
    binary_type: str,
) -> FloatArray:
    """Transform a binary feature column into its numeric representation.

    Maps each value to 0.0 or 1.0 based on the fitted mapping, and handles
    missing values (``NaN``) and unseen values according to ``handle_unseen``.

    Args:
        col (AnyArray): 1-D array of raw values for a single column.
        col_idx (int): Column index, used in error and warning messages.
        metadata (BinaryMetadata): Fitted metadata produced by
            :func:`~gower_metric.utils.binary_ut.fit_binary_features`, containing
            ``"mapping"``, ``"values"``, and ``"is_explicit_order"`` keys.
        handle_unseen (str): Strategy for unseen values — ``"error"``,
            ``"warning"``, or ``"missing"``.
        binary_type (str): The binary feature type name (e.g. ``"binary_asymmetric"``).

    Returns:
        FloatArray: 1-D float array of the same length as *col*, with values
        mapped to 0.0, 1.0, or ``np.nan``.

    Raises:
        ValueError: If an unseen value is encountered and *handle_unseen* is
            ``"error"``, or if an explicit value order is violated.

    """
    mapping = metadata["mapping"]
    is_explicit = metadata.get("is_explicit_order", False)

    non_null_mask = ~pd.isna(col)

    if not is_explicit:
        unique_transform_vals = set(np.unique(col[non_null_mask]))
        fitted_vals = set(mapping.keys())
        unseen_vals = unique_transform_vals - fitted_vals
        total_unique = len(fitted_vals) + len(unseen_vals)

        if total_unique > MAX_BINARY_UNIQUE_VALUES:
            short_type = binary_type.replace("binary_", "")
            msg = (
                f"Binary {short_type} column {col_idx} has {total_unique} unique values total "
                f"(fitted: {sorted(fitted_vals, key=str)}, "
                f"unseen: {sorted(unseen_vals, key=str)}). "
                f"Binary features must have at most {MAX_BINARY_UNIQUE_VALUES} values. "
                f"Consider using {binary_type}_value_order to explicitly define the expected binary values, "
                "or change the feature type if this is not actually a binary feature."
            )
            raise ValueError(msg)

    transformed_col = np.full(col.shape[0], np.nan, dtype=float)
    matched = ~non_null_mask
    for v, code in mapping.items():
        value_mask = non_null_mask & (col == v)
        transformed_col[value_mask] = code
        matched |= value_mask

    if not matched.all():
        for v in pd.unique(col[~matched]):
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

    return transformed_col


def transform_binary_asymmetric(
    col: AnyArray,
    col_idx: int,
    metadata: BinaryMetadata,
    handle_unseen: str,
) -> FloatArray:
    """Transform a binary asymmetric feature column into its numeric representation."""
    return _transform_binary(col, col_idx, metadata, handle_unseen, "binary_asymmetric")


def transform_binary_symmetric(
    col: AnyArray,
    col_idx: int,
    metadata: BinaryMetadata,
    handle_unseen: str,
) -> FloatArray:
    """Transform a binary symmetric feature column into its numeric representation."""
    return _transform_binary(col, col_idx, metadata, handle_unseen, "binary_symmetric")
