# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

import sys
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from gower_metric._typing import (
    DataFrameOrArray,
    FloatArray,
    FloatDType,
    MatrixType,
    NativeKernel,
)
from gower_metric.utils.cpp_middleware.config_builder import (
    _CLASSES_BY_ITEMSIZE,
    build_cpp_config,
)

if TYPE_CHECKING:
    from gower_metric import Gower

_OUT_DTYPES = [np.dtype(f"f{size}") for size in sorted(_CLASSES_BY_ITEMSIZE)]


def _as_native_matrix(model: "Gower", X: DataFrameOrArray) -> FloatArray:
    """Return a C-contiguous ``(n, n_features)`` array of ``model.data_type``.

    Args:
        model (gower_metric.Gower): fitted Gower instance.
        X (DataFrameOrArray): input data of shape (n_samples, n_features).

    Returns:
        FloatArray: contiguous numeric array ready for the native kernel.

    """
    arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)

    if not np.issubdtype(arr.dtype, np.number):
        with _temporary_skip_oor(model):  # the kernel checks the range
            transformed = model.transform(X)
        arr = (
            transformed.to_numpy()
            if isinstance(transformed, pd.DataFrame)
            else np.asarray(transformed)
        )

    return np.ascontiguousarray(arr, dtype=model.data_type)


def _validate_out(
    out: FloatArray,
    shape: tuple[int, ...],
) -> FloatArray:
    """Validate a user-supplied output buffer for zero-copy native writes.

    Args:
        out (FloatArray): user-supplied output buffer.
        shape (tuple[int, ...]): expected shape ``(rows, cols)``.

    Returns:
        FloatArray: the validated ``out`` (unchanged).

    Raises:
        TypeError: If ``out`` is not a NumPy array.
        ValueError: If shape, dtype, contiguity, or writability is wrong.

    """
    if not isinstance(out, np.ndarray):
        msg = "out must be a numpy ndarray"
        raise TypeError(msg)
    if out.shape != shape:
        msg = f"out must have shape {shape}, got {out.shape}"
        raise ValueError(msg)
    if out.dtype not in _OUT_DTYPES:
        msg = f"out must have dtype {', '.join(map(str, _OUT_DTYPES))}, got {out.dtype}"
        raise ValueError(msg)
    if not out.flags["C_CONTIGUOUS"]:
        msg = "out must be C-contiguous"
        raise ValueError(msg)
    if not out.flags["WRITEABLE"]:
        msg = "out must be writable"
        raise ValueError(msg)
    return out


@contextmanager
def _temporary_skip_oor(gower: "Gower") -> Generator[None]:
    """Temporarily disable out-of-range (OOR) checking on a Gower instance.

    This context manager sets `gower.skip_oor` to True for the duration of the
    context and restores its previous value afterward, even if an exception
    occurs.

    Args:
        gower (gower_metric.Gower): Fitted Gower instance.

    Yields:
        None

    Notes:
        This mutates shared state on the provided `gower` object. It is not
            thread-safe if the same instance is used concurrently in multiple computations.

    """
    old = gower.skip_oor
    gower.skip_oor = True
    try:
        yield
    finally:
        gower.skip_oor = old


def calculate_matrix(
    gower: "Gower",
    X: DataFrameOrArray,
    *,
    data_type: FloatDType | None = None,
    matrix_type: MatrixType = "distance",
    out: FloatArray | None = None,
    Y: DataFrameOrArray | None = None,
) -> FloatArray:
    """Return the pairwise Gower distance matrix.

    Scipy sparse matrices are not supported as direct input. The result is
    always dense; for a sparse one convert it, e.g. ``scipy.sparse.csr_matrix(m)``
    (scipy has no float16: convert with ``m.astype(np.float32)`` first).

    Args:
        gower (gower_metric.Gower): Fitted Gower instance.
        X (DataFrameOrArray): shape of (n_samples, n_features).
        data_type (FloatDType | None): data type used for the output distance matrix.
            If None, uses the data_type from the Gower instance configuration.
        matrix_type (str): Type of matrix to compute, either 'distance' or 'similarity'.
            Default is 'distance'.
        out (FloatArray | None): optional preallocated output buffer, written
            zero-copy. Its dtype may be any width the native kernel writes and
            need not match the model's ``data_type``; when given, it overrides
            the ``data_type`` argument. Default None.
        Y (DataFrameOrArray | None): optional second set; when given,
            returns the cross matrix X-vs-Y of shape (len(X), len(Y)). Default None.

    Returns:
        FloatArray:
            Pairwise Gower distance or similarity matrix of shape (n_samples, n_samples).

    Note:
        If fit(X) was not called before computing the matrix, the model will be
            fitted automatically and a UserWarning will be emitted.


    Examples:
        Basic usage:
            >>> import numpy as np
            >>> from gower_metric import Config, Gower
            >>> from gower_metric.utils.matrix.distance import calculate_matrix
            >>> data = np.array([[1, 'a'], [2, 'b'], [3, 'a'], [4, 'c']], dtype=object)
            >>> feature_types = {
            ...     0: 'ratio_scale_interval',
            ...     1: 'categorical_nominal',
            ... }
            >>> cfg = Config(
            ...     feature_types=feature_types,
            ... )
            >>> gower = Gower(cfg).fit(data)
            >>> distance_matrix = calculate_matrix(gower, data)
            >>> similarity_matrix = calculate_matrix(gower, data, matrix_type='similarity')

    Raises:
        ValueError: If ``X`` is a scipy sparse matrix or ``matrix_type`` is unknown.

    """
    sparse = sys.modules.get("scipy.sparse")  # loaded if X can be sparse at all
    if sparse is not None and sparse.issparse(X):
        msg = (
            "Sparse matrices are currently not supported as direct input. "
            "Please provide a dense matrix."
        )
        raise ValueError(msg)

    if not gower.is_fitted:
        gower.fit(X)
        msg = "Calling .fit(X) inside .matrix(X)."
        warnings.warn(msg, UserWarning, stacklevel=2)

    if data_type is None:
        data_type = gower.data_type

    if matrix_type not in ("distance", "similarity"):
        msg = (
            f"Unknown matrix_type '{matrix_type}'. Must be 'distance' or 'similarity'."
        )
        raise ValueError(msg)
    similarity = matrix_type == "similarity"

    data = _as_native_matrix(gower, X)
    data_y = None if Y is None else _as_native_matrix(gower, Y)
    shape = (len(data), len(data) if data_y is None else len(data_y))

    target = np.dtype(data_type)
    if target not in _OUT_DTYPES:
        target = np.dtype(gower.data_type)

    result = np.empty(shape, dtype=target) if out is None else _validate_out(out, shape)

    if gower.cpp_config is None:
        gower.cpp_config = build_cpp_config(gower)

    kernel = cast("NativeKernel", gower.cpp_config)
    if data_y is None:
        kernel.calculate_matrix(data, result, similarity)
    else:
        kernel.calculate_matrix_xy(data, data_y, result, similarity)

    if out is None and result.dtype != np.dtype(data_type):
        return result.astype(data_type)

    return result
