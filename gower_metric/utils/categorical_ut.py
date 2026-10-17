# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from gower_metric._typing import FittedEncoder, FloatDType, ObjectArray, ValueOrder


def _fit_encoder(
    column: ObjectArray,
    data_type: FloatDType,
    handle_unseen: str,
    categories: list[Sequence[object]] | None = None,
) -> FittedEncoder:
    """Build and fit the encoder for one column.

    Args:
        column (ObjectArray): The column's values, missing ones removed.
        data_type (FloatDType): Data type for the encoder output.
        handle_unseen (str): 'error' makes an unseen category an error; any
            other strategy encodes it as NaN for the caller to deal with.
        categories (list[Sequence[object]] | None): The declared levels in
            order, or None to learn the categories from the data.

    Returns:
        FittedEncoder: The fitted encoder.

    """
    options: dict[str, object] = {"dtype": data_type}
    if categories is not None:
        options["categories"] = categories
    if handle_unseen == "error":
        options["handle_unknown"] = "error"
    else:
        options["handle_unknown"] = "use_encoded_value"
        options["unknown_value"] = np.nan
    encoder = OrdinalEncoder(**options)  # type: ignore[no-untyped-call]
    encoder.fit(column.reshape(-1, 1))
    fitted: FittedEncoder = encoder
    return fitted


def fit_nominal_features(
    X: ObjectArray,
    nominal_indices: list[int],
    data_type: FloatDType = np.float32,
    handle_unseen: str = "error",
) -> dict[int, FittedEncoder]:
    """Fit OrdinalEncoders for categorical nominal features.

    Args:
        X (ObjectArray): The input data array.
        nominal_indices (list[int]): Indices of nominal columns.
        data_type (FloatDType): Data type for the encoder output.
        handle_unseen (str): Strategy for handling unseen categories ('error', 'warning', 'missing').

    Returns:
        dict[int, FittedEncoder]: A dictionary mapping column index to the fitted encoder.

    """
    nominal_metadata: dict[int, FittedEncoder] = {}
    for j in nominal_indices:
        col = X[:, j]
        col_clean = col[~pd.isna(col)]

        nominal_metadata[j] = _fit_encoder(col_clean, data_type, handle_unseen)
    return nominal_metadata


def fit_ordinal_features(
    X: ObjectArray,
    ordinal_indices: list[int],
    ordered_values: ValueOrder,
    data_type: FloatDType = np.float32,
    handle_unseen: str = "error",
) -> dict[int, FittedEncoder]:
    """Fit OrdinalEncoders for categorical ordinal features.

    Args:
        X (ObjectArray): The input data array.
        ordinal_indices (list[int]): Indices of ordinal columns.
        ordered_values (ValueOrder): Dictionary defining order for each column.
        data_type (FloatDType): Data type for the encoder output.
        handle_unseen (str): Strategy for handling unseen categories ('error', 'warning', 'missing').

    Returns:
        dict[int, FittedEncoder]: A dictionary mapping column index to the fitted encoder.

    Raises:
        ValueError: If ordered_values is missing or incomplete.

    """
    ordinal_metadata: dict[int, FittedEncoder] = {}
    for j in ordinal_indices:
        col = X[:, j]
        if ordered_values is None:  # pragma: no cover
            msg = "Categorical ordinal values order is missing"
            raise ValueError(msg)

        col_clean = col[~pd.isna(col)]

        ordinal_metadata[j] = _fit_encoder(
            col_clean,
            data_type,
            handle_unseen,
            [ordered_values[j]],
        )
    return ordinal_metadata
