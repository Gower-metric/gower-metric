# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

from collections.abc import Sequence

import numpy as np
import pandas as pd

from gower_metric._typing import BinaryMetadata, ObjectArray, ValueOrder

MAX_BINARY_UNIQUE_VALUES = 2


def _is_presence_coding(values: object) -> bool:
    """Whether every value is 0/1 or False/True, so presence is unambiguous."""
    vals = list(np.asarray(values, dtype=object).ravel())
    if any(isinstance(v, (str, bytes)) for v in vals):
        return False
    try:
        return all(v in (0, 1) for v in vals)
    except TypeError:  # pragma: no cover - unorderable objects
        return False


def fit_binary_features(
    arr: ObjectArray,
    binary_indices: list[int],
    binary_value_order: ValueOrder = None,
    *,
    asymmetric: bool = False,
    col_names: Sequence[object] | None = None,
) -> dict[int, BinaryMetadata]:
    """Fit binary features metadata.

    Args:
        arr (ObjectArray): Input data array.
        binary_indices (list[int]): Indices of binary features.
        binary_value_order (ValueOrder): Optional explicit ordering of binary values.
            If provided, specifies expected values for each binary column (must be exactly 2 values).
            Creates full mapping for all expected values, even if not present in training data.
            If None, values are auto-detected from training data.
        asymmetric (bool): Whether these are asymmetric (presence/absence) features.
        col_names (Sequence[object] | None): Column names, for error messages.

    Returns:
        dict[int, BinaryMetadata]: Metadata for binary features, including:
            - mapping: dict mapping values to 0.0/1.0
            - is_explicit_order: bool indicating if explicit order was used

    Raises:
        ValueError: If a binary column has more than 2 unique values, or if training data
            contains values not in binary_value_order when explicit ordering is used.

    """
    binary_metadata: dict[int, BinaryMetadata] = {}

    for j in binary_indices:
        col = arr[:, j]
        non_null_mask = ~pd.isna(col)
        valid_values = col[non_null_mask]

        if binary_value_order is not None and j in binary_value_order:
            expected_vals = binary_value_order[j]

            actual_unique = pd.unique(valid_values)
            unexpected = set(actual_unique) - set(expected_vals)
            if unexpected:
                msg = (
                    f"Binary feature at index {j} contains unexpected values: "
                    f"{unexpected}. Expected values: {expected_vals}"
                )
                raise ValueError(msg)

            mapping = {expected_vals[0]: 0.0, expected_vals[1]: 1.0}

            binary_metadata[j] = {"mapping": mapping, "is_explicit_order": True}
            continue

        unique_vals = np.sort(pd.unique(valid_values))
        if len(unique_vals) > MAX_BINARY_UNIQUE_VALUES:
            msg = (
                f"Binary feature at index {j} has more than {MAX_BINARY_UNIQUE_VALUES} "
                f"unique values: {unique_vals}. "
                f"Gower metric expects at most {MAX_BINARY_UNIQUE_VALUES} "
                "unique values for binary features."
            )
            raise ValueError(msg)

        if asymmetric:
            if not _is_presence_coding(unique_vals):
                name = repr(col_names[j]) if col_names is not None else f"at index {j}"
                observed = sorted({str(v) for v in valid_values})
                msg = (
                    f"Asymmetric binary feature {name} is coded as {observed}, so it "
                    "is not clear which value means 'present'."
                )
                raise ValueError(msg)
            binary_metadata[j] = {
                "mapping": {0: 0.0, 1: 1.0},
                "is_explicit_order": False,
            }

        else:
            mapping = {}
            if len(unique_vals) > 0:
                mapping[unique_vals[0]] = 0.0
            if len(unique_vals) > 1:
                mapping[unique_vals[1]] = 1.0

            binary_metadata[j] = {"mapping": mapping, "is_explicit_order": False}

    return binary_metadata
