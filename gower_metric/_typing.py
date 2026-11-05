# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

"""Shared type aliases."""

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

ObjectArray: TypeAlias = npt.NDArray[np.object_]
"""Raw records or columns."""

FloatArray: TypeAlias = npt.NDArray[np.floating]
"""Encoded values in any float width."""

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
    is_explicit_order: bool


class OrdinalMetadata(TypedDict):
    """What ``fit`` learns about one ordinal column: the count of each level."""

    counts: FloatArray


class FittedEncoder(Protocol):
    """What the package reads from a fitted scikit-learn ``OrdinalEncoder``."""

    @property
    def categories_(self) -> Sequence[AnyArray]:
        """The fitted categories, one array per encoded column."""
        ...

    @property
    def handle_unknown(self) -> str:
        """The ``handle_unknown`` setting the encoder was built with."""
        ...


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


MatrixType: TypeAlias = Literal["distance", "similarity"]
"""Which of the two complementary matrices to compute."""

__all__ = [
    "AnyArray",
    "BinaryMetadata",
    "DataFrameOrArray",
    "FittedEncoder",
    "FloatArray",
    "FloatDType",
    "MatrixType",
    "NativeKernel",
    "ObjectArray",
    "OrdinalMetadata",
    "Record",
    "RecordKey",
    "Transform",
    "ValueOrder",
]
