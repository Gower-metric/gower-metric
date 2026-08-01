import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

MAP_THRESHOLD = 3000
"""Row count above which ``pandas.Series.map`` overtakes a Python-level lookup."""

_MAPPING_ATTR = "_gower_code_mapping"


def _code_mapping(enc: OrdinalEncoder) -> dict:
    """Return the fitted category-to-code mapping.

    Args:
        enc (OrdinalEncoder): Encoder fitted on a single column.

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
        mapping = {
            category: float(code) for code, category in enumerate(enc.categories_[0])
        }
        setattr(enc, _MAPPING_ATTR, mapping)
    return mapping


def encode_categories(
    values: np.ndarray,
    enc: OrdinalEncoder,
    data_type: type[np.floating],
) -> np.ndarray:
    """Encode already-non-missing values.

    Args:
        values (np.ndarray): 1-D array of raw values, guaranteed free of
            missing entries by the caller.
        enc (OrdinalEncoder): Encoder fitted on this column.
        data_type (type[np.floating]): Output precision.

    Returns:
        np.ndarray: 1-D codes, with ``np.nan`` where a value was not fitted.

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
            msg = f"Found unknown categories {found} in column 0 during transform"
            raise ValueError(msg)

    return codes
