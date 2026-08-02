# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from gower_metric._typing import AnyArray, FloatArray, FloatDType
from gower_metric.utils.transforms.encoding import encode_categories


def transform_categorical_nominal(
    col: AnyArray,
    col_idx: int,
    enc: OrdinalEncoder,
    handle_unseen: str,
    data_type: FloatDType,
) -> FloatArray:
    """Transform a categorical nominal feature column.

    Args:
        col (AnyArray): The column data to transform.
        col_idx (int): The column index (for error messages).
        enc (OrdinalEncoder): The fitted OrdinalEncoder.
        handle_unseen (str): Strategy for unseen values ('error', 'warning', 'missing').
        data_type: NumPy data type for the output array.

    Returns:
        FloatArray: Transformed column with encoded values.

    """
    col_arr = np.array(col)
    non_null_mask = ~pd.isna(col_arr)
    transformed_col = np.full(col_arr.shape[0], np.nan, dtype=data_type)

    if non_null_mask.any():
        transformed_col[non_null_mask] = encode_categories(
            col_arr[non_null_mask],
            enc,
            data_type,
        )

    if handle_unseen == "warning":
        nan_output = np.isnan(transformed_col)
        unseen_mask = non_null_mask & nan_output
        if unseen_mask.any():
            unseen_vals = sorted(set(col_arr[unseen_mask]))
            warnings.warn(
                f"Unseen values {unseen_vals} in nominal column {col_idx} "
                f"not found in fitted categories. Treating as missing (np.nan).",
                UserWarning,
                stacklevel=2,
            )

    return transformed_col


def transform_categorical_ordinal(
    col: AnyArray,
    col_idx: int,
    enc: OrdinalEncoder,
    handle_unseen: str,
    data_type: FloatDType,
) -> FloatArray:
    """Transform a categorical ordinal feature column.

    Args:
        col (AnyArray): The column data to transform.
        col_idx (int): The column index (for error messages).
        enc (OrdinalEncoder): The fitted OrdinalEncoder.
        handle_unseen (str): Strategy for unseen values ('error', 'warning', 'missing').
        data_type: NumPy data type for the output array.

    Returns:
        FloatArray: Transformed column with encoded values.

    """
    col_arr = np.array(col)
    non_null_mask = ~pd.isna(col_arr)
    transformed_col = np.full(col_arr.shape[0], np.nan, dtype=data_type)

    if non_null_mask.any():
        transformed_col[non_null_mask] = encode_categories(
            col_arr[non_null_mask],
            enc,
            data_type,
        )

    if handle_unseen == "warning":
        nan_output = np.isnan(transformed_col)
        unseen_mask = non_null_mask & nan_output
        if unseen_mask.any():
            unseen_vals = sorted(set(col_arr[unseen_mask]))
            warnings.warn(
                f"Unseen values {unseen_vals} in ordinal column {col_idx} "
                f"not found in fitted categories. Treating as missing (np.nan).",
                UserWarning,
                stacklevel=2,
            )

    return transformed_col
