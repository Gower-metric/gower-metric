# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from gower_metric._typing import AnyArray, FittedEncoder, FloatArray, FloatDType

if TYPE_CHECKING:
    from gower_metric.core.metric import Gower

MAP_THRESHOLD = 3000
"""Row count above which ``pandas.Series.map`` overtakes a Python-level lookup."""

_MAPPING_ATTR = "_gower_code_mapping"
_NUMBERS = (int, float, np.integer, np.floating, np.bool_)


def own_codes(values: Iterable[object]) -> dict[object, float] | None:
    """Code a column of numbers as itself; None if a value is not a number."""
    codes: dict[object, float] = {}
    for value in values:
        if not isinstance(value, _NUMBERS):
            return None
        codes[value] = float(value)
    return codes


def _code_mapping(enc: FittedEncoder) -> dict[object, float]:
    """Return the fitted category-to-code mapping.

    Args:
        enc (FittedEncoder): Encoder fitted on a single column.

    Returns:
        dict: Mapping from fitted category to its integer code as a float.

    Note:
        Values are plain Python floats, not the model's ``data_type``. The
        mapping is memoised per encoder and the output precision is a property
        of the call, so keeping it dtype-agnostic lets one encoder serve any
        requested precision; the conversion happens when the codes array is
        built.

    """
    mapping = getattr(enc, _MAPPING_ATTR, None)
    if mapping is None:
        mapping = own_codes(enc.categories_[0]) or {
            category: float(code) for code, category in enumerate(enc.categories_[0])
        }
        setattr(enc, _MAPPING_ATTR, mapping)
    return mapping


def encode_categories(
    values: AnyArray,
    enc: FittedEncoder,
    data_type: FloatDType,
    col_idx: int,
) -> FloatArray:
    """Encode already-non-missing values.

    Args:
        values (AnyArray): 1-D array of raw values, guaranteed free of
            missing entries by the caller.
        enc (FittedEncoder): Encoder fitted on this column.
        data_type (FloatDType): Output precision.
        col_idx (int): Column index, named in the error message.

    Returns:
        FloatArray: 1-D codes, with ``np.nan`` where a value was not fitted.

    Raises:
        ValueError: If the encoder was fitted with ``handle_unknown="error"``
            and an unfitted value is present.

    """
    mapping = _code_mapping(enc)
    count = values.shape[0]

    codes = (
        np.fromiter(
            (mapping.get(value, np.nan) for value in values),
            dtype=data_type,
            count=count,
        )
        if count < MAP_THRESHOLD
        else pd.Series(values).map(mapping).to_numpy(dtype=data_type)
    )

    if enc.handle_unknown == "error":
        unknown = np.isnan(codes)
        if unknown.any():
            found = list(pd.unique(values[unknown]))
            msg = (
                f"Found unknown categories {found} in column {col_idx} during transform"
            )
            raise ValueError(msg)

    return codes


def fitted_codes(model: "Gower") -> dict[int, dict[object, float]]:
    """Give the code of each fitted value of every nominal and binary feature."""
    binary = {**model.binary_symmetric_metadata, **model.binary_asymmetric_metadata}
    return {
        **{j: _code_mapping(e) for j, e in model.nominal_metadata.items()},
        **{j: own_codes(b["mapping"]) or b["mapping"] for j, b in binary.items()},
    }
