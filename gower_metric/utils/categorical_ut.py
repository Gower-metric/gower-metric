from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def fit_nominal_features(
    X: np.ndarray,
    nominal_indices: list[int],
    data_type: type[np.floating | np.integer] = np.float32,
    handle_unseen: str = "error",
) -> dict[int, OrdinalEncoder]:
    """Fit OrdinalEncoders for categorical nominal features.

    Args:
        X (np.ndarray): The input data array.
        nominal_indices (list[int]): Indices of nominal columns.
        data_type (type[np.floating | np.integer]): Data type for the encoder output.
        handle_unseen (str): Strategy for handling unseen categories ('error', 'warning', 'missing').

    Returns:
        dict[int, OrdinalEncoder]: A dictionary mapping column index to the fitted encoder.

    """
    nominal_metadata: dict[int, OrdinalEncoder] = {}
    for j in nominal_indices:
        col = X[:, j]
        col_clean = col[~pd.isna(col)]

        enc = (
            OrdinalEncoder(dtype=data_type, handle_unknown="error")
            if handle_unseen == "error"
            else OrdinalEncoder(
                dtype=data_type,
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
            )
        )
        enc.fit(col_clean.reshape(-1, 1))
        nominal_metadata[j] = enc
    return nominal_metadata


def fit_ordinal_features(
    X: np.ndarray,
    ordinal_indices: list[int],
    ordered_values: dict[int | str, list[Any]] | None,
    data_type: type[np.floating | np.integer] = np.float32,
    handle_unseen: str = "error",
) -> dict[int, OrdinalEncoder]:
    """Fit OrdinalEncoders for categorical ordinal features.

    Args:
        X (np.ndarray): The input data array.
        ordinal_indices (list[int]): Indices of ordinal columns.
        ordered_values (dict[int | str, list[Any]] | None): Dictionary defining order for each column.
        data_type (type[np.floating | np.integer]): Data type for the encoder output.
        handle_unseen (str): Strategy for handling unseen categories ('error', 'warning', 'missing').

    Returns:
        dict[int, OrdinalEncoder]: A dictionary mapping column index to the fitted encoder.

    Raises:
        ValueError: If ordered_values is missing or incomplete.

    """
    ordinal_metadata: dict[int, OrdinalEncoder] = {}
    for j in ordinal_indices:
        col = X[:, j]
        if ordered_values is None:  # pragma: no cover
            msg = "Categorical ordinal values order is missing"
            raise ValueError(msg)

        col_clean = col[~pd.isna(col)]

        if handle_unseen == "error":
            enc = OrdinalEncoder(
                categories=[ordered_values[j]],
                dtype=data_type,
                handle_unknown="error",
            )
        else:
            enc = OrdinalEncoder(
                categories=[ordered_values[j]],
                dtype=data_type,
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
            )
        enc.fit(col_clean.reshape(-1, 1))
        ordinal_metadata[j] = enc
    return ordinal_metadata
