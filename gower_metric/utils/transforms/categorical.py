import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def transform_categorical_nominal(
    col: np.ndarray,
    col_idx: int,
    enc: OrdinalEncoder,
    handle_unseen: str,
    data_type: type[np.floating | np.integer],
) -> np.ndarray:
    """Transform a categorical nominal feature column.

    Args:
        col (np.ndarray): The column data to transform.
        col_idx (int): The column index (for error messages).
        enc (OrdinalEncoder): The fitted OrdinalEncoder.
        handle_unseen (str): Strategy for unseen values ('error', 'warning', 'missing').
        data_type: NumPy data type for the output array.

    Returns:
        np.ndarray: Transformed column with encoded values.

    """
    col_arr = np.array(col)
    non_null_mask = ~pd.isna(col_arr)
    transformed_col = np.full(col_arr.shape[0], np.nan, dtype=data_type)

    if non_null_mask.any():
        encoded = (
            enc.transform(col_arr[non_null_mask].reshape(-1, 1))
            .astype(data_type)
            .ravel()
        )
        transformed_col[non_null_mask] = encoded

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
    col: np.ndarray,
    col_idx: int,
    enc: OrdinalEncoder,
    handle_unseen: str,
    data_type: type[np.floating | np.integer],
) -> np.ndarray:
    """Transform a categorical ordinal feature column.

    Args:
        col (np.ndarray): The column data to transform.
        col_idx (int): The column index (for error messages).
        enc (OrdinalEncoder): The fitted OrdinalEncoder.
        handle_unseen (str): Strategy for unseen values ('error', 'warning', 'missing').
        data_type: NumPy data type for the output array.

    Returns:
        np.ndarray: Transformed column with encoded values.

    """
    col_arr = np.array(col)
    non_null_mask = ~pd.isna(col_arr)
    transformed_col = np.full(col_arr.shape[0], np.nan, dtype=data_type)

    if non_null_mask.any():
        encoded = (
            enc.transform(col_arr[non_null_mask].reshape(-1, 1))
            .astype(data_type)
            .ravel()
        )
        transformed_col[non_null_mask] = encoded

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
