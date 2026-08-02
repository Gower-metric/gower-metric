"""Shared type aliases."""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

ObjectArray: TypeAlias = npt.NDArray[np.object_]
"""Raw records or columns."""

FloatArray: TypeAlias = npt.NDArray[np.floating]
"""Encoded values in any float width."""

BoolArray: TypeAlias = npt.NDArray[np.bool_]
"""Element-wise masks over a column."""

AnyArray: TypeAlias = npt.NDArray[np.generic]
"""An array of any dtype."""

FloatDType: TypeAlias = type[np.floating]
"""A float precision selected by configuration."""

DataFrameOrArray: TypeAlias = pd.DataFrame | AnyArray
"""A tabular input accepted at the public boundary."""

Record: TypeAlias = AnyArray | pd.Series | Sequence[object]
"""A single observation."""

RecordKey: TypeAlias = tuple[object, ...]
"""A record identified by its values, for use as a mapping key."""

Transform: TypeAlias = Callable[[ObjectArray], pd.DataFrame | FloatArray]
"""Encoder mapping a block of raw values to its numeric representation."""

ValueOrder: TypeAlias = Mapping[int | str, Sequence[object]] | None
"""User-declared ordering of a column's levels, keyed by index or name."""


class BinaryMetadata(TypedDict):
    """What ``fit_binary_features`` learns about one binary column."""

    mapping: dict[object, float]
    values: ObjectArray
    is_explicit_order: bool
    positive_value: object


class OrdinalMetadata(TypedDict):
    """What ``fit`` learns about one ordinal column's levels and cardinalities."""

    ranks: dict[object, int]
    denom: float
    counts: FloatArray
    min: int | None
    max: int | None


class NativeKernel(Protocol):
    """The compute entry points shared by the native config classes."""

    def calculate_distance(self, x: FloatArray, y: FloatArray) -> float:
        """Return the Gower distance between two encoded records."""
        ...

    def calculate_matrix(
        self,
        data: FloatArray,
        out: FloatArray,
        similarity: bool,
    ) -> None:
        """Fill ``out`` with the pairwise matrix of ``data``."""
        ...

    def calculate_matrix_xy(
        self,
        x_data: FloatArray,
        y_data: FloatArray,
        out: FloatArray,
        similarity: bool,
    ) -> None:
        """Fill ``out`` with the cross matrix of ``x_data`` against ``y_data``."""
        ...


__all__ = [
    "AnyArray",
    "BinaryMetadata",
    "BoolArray",
    "DataFrameOrArray",
    "FloatArray",
    "FloatDType",
    "NativeKernel",
    "ObjectArray",
    "OrdinalMetadata",
    "Record",
    "RecordKey",
    "Transform",
    "ValueOrder",
]
