import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse

from gower_metric.utils.cpp_middleware.config_builder import build_cpp_config
from gower_metric.utils.deprecation import retired_args
from gower_metric.utils.matrix.convert_matrix import get_scipy_sparse_matrix
from gower_metric.utils.ranges import enforce_oor_policy

if TYPE_CHECKING:
    from gower_metric import Gower

_RETIRED_PARAMS_REASON = "no longer has any effect."


def _as_native_matrix(model: "Gower", X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return a C-contiguous ``(n, n_features)`` array of ``model.data_type``.

    Args:
        model (gower_metric.Gower): fitted Gower instance.
        X (pd.DataFrame | np.ndarray): input data of shape (n_samples, n_features).

    Returns:
        np.ndarray: contiguous numeric array ready for the native kernel.

    """
    arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)

    if not np.issubdtype(arr.dtype, np.number):
        transformed = model.transform(X)
        arr = (
            transformed.to_numpy()
            if isinstance(transformed, pd.DataFrame)
            else np.asarray(transformed)
        )

    return np.ascontiguousarray(arr, dtype=model.data_type)


def _validate_out(
    out: np.ndarray,
    shape: tuple[int, int],
    data_type: type[np.floating],
) -> np.ndarray:
    """Validate a user-supplied output buffer for zero-copy native writes.

    Args:
        out (np.ndarray): user-supplied output buffer.
        shape (tuple[int, int]): expected ``(rows, cols)`` shape.
        data_type (type[np.floating]): expected dtype (the native compute precision).

    Returns:
        np.ndarray: the validated ``out`` (unchanged).

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
    if out.dtype != np.dtype(data_type):
        msg = (
            f"out must have dtype {np.dtype(data_type)} (the model's data_type), "
            f"got {out.dtype}"
        )
        raise ValueError(msg)
    if not out.flags["C_CONTIGUOUS"]:
        msg = "out must be C-contiguous"
        raise ValueError(msg)
    if not out.flags["WRITEABLE"]:
        msg = "out must be writable"
        raise ValueError(msg)
    return out


def _get_full_matrix(
    gower: "Gower",
    X: pd.DataFrame | np.ndarray,
    data_type: type[np.floating],
    matrix_type: str = "distance",
    out: np.ndarray | None = None,
    convert_to_sparse: bool = False,
    sparse_type: str = "csr",
    Y: pd.DataFrame | np.ndarray | None = None,
) -> (
    np.ndarray
    | scipy.sparse.csr_matrix
    | scipy.sparse.csc_matrix
    | scipy.sparse.coo_matrix
):
    """Compute the pairwise Gower matrix in the native kernel.

    Args:
        gower (gower_metric.Gower): fitted Gower instance.
        X (pd.DataFrame | np.ndarray): shape (n_samples, n_features).
        data_type (type[np.floating]): dtype of the returned matrix. The kernel
            computes in ``gower.data_type``; the result is cast at the end if it
            differs (only when ``out`` is not supplied).
        matrix_type (str): 'distance' or 'similarity'. Default 'distance'.
            Converted to a flag the kernel applies per element; no Python-side
            post-processing takes place.
        out (np.ndarray | None): optional preallocated output buffer of
            ``gower.data_type``; written in place (zero-copy). Default None.
        convert_to_sparse (bool): convert the dense result to sparse. Default False.
        sparse_type (str): 'csr', 'csc' or 'coo'. Default 'csr'.
        Y (pd.DataFrame | np.ndarray | None): optional second set; when given a
            cross matrix X-vs-Y is computed. Default None.

    Returns:
        np.ndarray | scipy.sparse matrix: the pairwise matrix.

    Raises:
        ValueError: For an unknown ``matrix_type`` or an ``out``/``convert_to_sparse``
            combination.

    """
    if matrix_type not in ("distance", "similarity"):
        msg = (
            f"Unknown matrix_type '{matrix_type}'. Must be 'distance' or 'similarity'."
        )
        raise ValueError(msg)
    similarity = matrix_type == "similarity"

    if out is not None and convert_to_sparse:
        msg = "out cannot be combined with convert_to_sparse"
        raise ValueError(msg)

    data = _as_native_matrix(gower, X)
    data_y = None if Y is None else _as_native_matrix(gower, Y)
    shape = (data.shape[0], data.shape[0] if data_y is None else data_y.shape[0])

    result = (
        np.empty(shape, dtype=gower.data_type)
        if out is None
        else _validate_out(out, shape, gower.data_type)
    )

    if gower.cpp_config is None:
        gower.cpp_config = build_cpp_config(gower)

    if data_y is None:
        gower.cpp_config.calculate_matrix(data, result, similarity)
    else:
        gower.cpp_config.calculate_matrix_xy(data, data_y, result, similarity)

    if convert_to_sparse:
        return get_scipy_sparse_matrix(
            result,
            matrix_format=sparse_type,
            data_type=data_type,
        )

    if out is None and np.dtype(data_type) != np.dtype(gower.data_type):
        return result.astype(data_type)

    return result


@contextmanager
def _temporary_skip_oor(gower: "Gower") -> Iterator[None]:
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


@retired_args(_RETIRED_PARAMS_REASON, "n_jobs", "verbose", "backend")
def calculate_matrix(
    gower: "Gower",
    X: pd.DataFrame | np.ndarray,
    *,
    data_type: type[np.floating] | None = None,
    matrix_type: str = "distance",
    convert_to_sparse: bool = False,
    sparse_type: str = "csr",
    out: np.ndarray | None = None,
    Y: pd.DataFrame | np.ndarray | None = None,
) -> (
    np.ndarray
    | scipy.sparse.csr_matrix
    | scipy.sparse.csc_matrix
    | scipy.sparse.coo_matrix
):
    """Return the pairwise Gower distance matrix.

    Scipy sparse matrices are not supported as direct input.

    Args:
        gower (gower_metric.Gower): Fitted Gower instance.
        X (pd.DataFrame | np.ndarray): shape of (n_samples, n_features).
        data_type (type[np.floating] | None): data type used for the output distance matrix.
            If None, uses the data_type from the Gower instance configuration.
        matrix_type (str): Type of matrix to compute, either 'distance' or 'similarity'.
            Default is 'distance'.
        convert_to_sparse (bool): Whether to convert the output dense matrix to a sparse format.
            Default is False.
        sparse_type (str): Type of sparse matrix to convert to, either 'csr', 'csc' or 'coo'.
            Default is 'csr'.
        out (np.ndarray | None): optional preallocated output buffer of the model's
            data_type, written zero-copy. Default None.
        Y (pd.DataFrame | np.ndarray | None): optional second set; when given,
            returns the cross matrix X-vs-Y of shape (len(X), len(Y)). Default None.

    Returns:
        np.ndarray | scipy.sparse.csr_matrix | scipy.sparse.csc_matrix | scipy.sparse.coo_matrix:
            Pairwise Gower distance or similarity matrix of shape (n_samples, n_samples) or sparse matrix.

    Warns:
        DeprecationWarning: If ``n_jobs``, ``verbose`` or ``backend`` is passed,
            since it is discarded.

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

        Sparse output:
            >>> sparse_similarity_matrix = calculate_matrix(
            ...     gower,
            ...     data,
            ...     matrix_type='similarity',
            ...     convert_to_sparse=True,
            ...     sparse_type='coo',
            ... )

    Raises:
        ValueError: If ``X`` is a scipy sparse matrix.

    """
    if scipy.sparse.issparse(X):
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

    if not gower.skip_oor:
        for operand in (X,) if Y is None else (X, Y):
            enforce_oor_policy(
                operand,
                strategy=gower.out_of_range,
                numeric_indices=gower.numeric_indices,
                numeric_mins=gower.numeric_mins,
                numeric_maxs=gower.numeric_maxs,
                ratio_scale_indices=gower.ratio_scale_indices,
                ratio_mins=gower.ratio_mins,
                ratio_maxs=gower.ratio_maxs,
                stacklevel=2,
            )

    with _temporary_skip_oor(gower):
        return _get_full_matrix(
            gower,
            X,
            data_type=data_type,
            matrix_type=matrix_type,
            out=out,
            convert_to_sparse=convert_to_sparse,
            sparse_type=sparse_type,
            Y=Y,
        )
