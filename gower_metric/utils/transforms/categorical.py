# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

import warnings

import numpy as np
import pandas as pd

from gower_metric._typing import AnyArray, FittedEncoder, FloatArray, FloatDType
from gower_metric.utils.transforms.encoding import encode_categories


def _transform_column(
    col: AnyArray,
    col_idx: int,
    enc: FittedEncoder,
    handle_unseen: str,
    data_type: FloatDType,
    kind: str,
) -> FloatArray:
    """Encode a categorical column, warning about unseen values if asked to."""
    col_arr = np.array(col)
    non_null_mask = ~pd.isna(col_arr)
    transformed_col = np.full(col_arr.shape[0], np.nan, dtype=data_type)

    if non_null_mask.any():
        transformed_col[non_null_mask] = encode_categories(
            col_arr[non_null_mask],
            enc,
            data_type,
            col_idx,
        )

    if handle_unseen == "warning":
        nan_output = np.isnan(transformed_col)
        unseen_mask = non_null_mask & nan_output
        if unseen_mask.any():
            unseen_vals = sorted(set(col_arr[unseen_mask]))
            warnings.warn(
                f"Unseen values {unseen_vals} in {kind} column {col_idx} "
                f"not found in fitted categories. Treating as missing (np.nan).",
                UserWarning,
                stacklevel=3,
            )

    return transformed_col


def transform_categorical_nominal(
    col: AnyArray,
    col_idx: int,
    enc: FittedEncoder,
    handle_unseen: str,
    data_type: FloatDType,
) -> FloatArray:
    """Transform a categorical nominal feature column.

    Args:
        col (AnyArray): The column data to transform.
        col_idx (int): The column index (for error messages).
        enc (FittedEncoder): The fitted OrdinalEncoder.
        handle_unseen (str): Strategy for unseen values ('error', 'warning', 'missing').
        data_type: NumPy data type for the output array.

    Returns:
        FloatArray: Transformed column with encoded values.

    """
    return _transform_column(col, col_idx, enc, handle_unseen, data_type, "nominal")


def transform_categorical_ordinal(
    col: AnyArray,
    col_idx: int,
    enc: FittedEncoder,
    handle_unseen: str,
    data_type: FloatDType,
) -> FloatArray:
    """Transform a categorical ordinal feature column.

    Args:
        col (AnyArray): The column data to transform.
        col_idx (int): The column index (for error messages).
        enc (FittedEncoder): The fitted OrdinalEncoder.
        handle_unseen (str): Strategy for unseen values ('error', 'warning', 'missing').
        data_type: NumPy data type for the output array.

    Returns:
        FloatArray: Transformed column with encoded values.

    """
    return _transform_column(col, col_idx, enc, handle_unseen, data_type, "ordinal")
