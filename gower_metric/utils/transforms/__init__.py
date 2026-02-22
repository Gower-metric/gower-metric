from gower_metric.utils.transforms.binary import (
    transform_binary_asymmetric,
    transform_binary_symmetric,
)
from gower_metric.utils.transforms.categorical import (
    transform_categorical_nominal,
    transform_categorical_ordinal,
)

__all__ = [
    "transform_binary_asymmetric",
    "transform_binary_symmetric",
    "transform_categorical_nominal",
    "transform_categorical_ordinal",
]

__doc__ = """
Transform helpers for converting raw feature columns into numeric representations.

These functions are called internally by :meth:`Gower.transform` to encode
binary and categorical columns. Each function respects the ``handle_unseen``
strategy configured via :class:`Config`.
"""
