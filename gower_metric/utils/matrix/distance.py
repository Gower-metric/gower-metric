from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse
from contextlib import contextmanager
from gower_metric.utils.ranges import enforce_oor_policy
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from gower_metric.utils.matrix.convert_matrix import get_scipy_sparse_matrix

if TYPE_CHECKING:
    from gower_metric import Gower


def __compute_row_upper(
    i: int,
    X_arr: np.ndarray,
    n: int,
    model: "Gower",
    data_type: type[np.floating],
    row_type: str,
) -> tuple[int, np.ndarray]:
    """Compute one upper triangle row of Gower distances.

    Args:
        i (int): row index.
        X_arr (np.ndarray): data array of shape (n_samples, n_features).
        n (int): number of samples.
        model (gower_metric.Gower): fitted Gower instance.
        data_type (type[np.floating]): data type for the output row array.
        row_type (str): type of row to compute, distance or similarity. Defaults to "distance".

    Returns:
        tuple[i, row]: tuple of row index and computed row array.

    """
    n = X_arr.shape[0] if n == 0 else n
    xi = X_arr[i]

    start = i + 1
    count = n - start

    if count <= 0:
        return (i, np.empty(0, dtype=data_type))

    func = model.similarity if row_type == "similarity" else model

    values = (func(xi, X_arr[j]) for j in range(start, n))

    row = np.fromiter(values, dtype=data_type, count=count)

    return (i, row)


def _get_results_from_joblib(
    arr: np.ndarray,
    n_jobs: int,
    verbose: int,
    data_type: type[np.floating],
    model: "Gower",
    matrix_type: str,
    backend: str = "loky",
    n: int = 0,
) -> list[tuple[int, np.ndarray]]:
    """Get results from joblib parallel processing.

    Args:
        arr (np.ndarray): data array of shape (n_samples, n_features).
        n_jobs (int): number of parallel jobs.
        verbose (int): whether to show progress bar.
        data_type (type[np.floating]): data type for the output rows.
        model (gower_metric.Gower): fitted Gower instance.
        matrix_type (str): type of matrix to compute, distance or similarity. Defaults to "distance".
        backend (str): joblib backend to use. Defaults to "loky".
        n (int): number of samples (if 0, will be set to arr.shape[0]).

    Returns:
        list[tuple[int, np.ndarray]]: List of tuples (row index, computed row array).

    """
    results: list[tuple[int, np.ndarray]] = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
    )(
        delayed(__compute_row_upper)(i, arr, n, model, data_type, matrix_type)
        for i in tqdm(
            range(n),
            desc="Calculating upper triangle rows",
            unit="row",
            disable=not verbose,
        )
    )

    return results


def _get_full_matrix(
    gower: "Gower",
    X: pd.DataFrame | np.ndarray,
    data_type: type[np.floating],
    n_jobs: int = -1,
    verbose: int = 0,
    matrix_type: str = "distance",
    convert_to_sparse: bool = False,
    sparse_type: str = "csr",
    backend: str = "loky",
) -> (
    np.ndarray
    | scipy.sparse.csr_matrix
    | scipy.sparse.csc_matrix
    | scipy.sparse.coo_matrix
):
    """Compute symmetric pairwise Gower distance matrix using joblib (parallel).

    Args:
        gower (gower_metric.Gower): Fitted Gower instance.
        X (pd.DataFrame | np.ndarray): shape of (n_samples, n_features).
        data_type (type[np.floating]): data type for the output distance matrix, default gower.data_type.
        n_jobs (int): number of parallel jobs to run, -1 means using all processors. Default is -1.
        verbose (int): whether to show tqdm progress bar. Default is 0 (no progress bar).
        matrix_type (str): Type of matrix to compute, either 'distance' or 'similarity'.
            Default is 'distance'.
        convert_to_sparse (bool): Whether to convert the output dense matrix to a sparse format.
            Default is False.
        sparse_type (str): Type of sparse matrix to convert to, either 'csr', 'csc' or 'coo'.
            Default is 'csr'.
        backend (str): Backend to use for joblib parallelization. Default is 'loky'.

    Returns:
        np.ndarray | scipy.sparse.csr_matrix | scipy.sparse.csc_matrix | scipy.sparse.coo_matrix:
            Pairwise Gower distance or similarity matrix of shape (n_samples, n_samples) or sparse matrix.

    """
    arr = (
        X.to_numpy(dtype=object)
        if isinstance(X, pd.DataFrame)
        else np.array(X, dtype=object)
    )

    n: int = arr.shape[0]

    result_matrix: np.ndarray = np.zeros((n, n), dtype=data_type)

    results: list[tuple[int, np.ndarray]] = _get_results_from_joblib(
        n_jobs=n_jobs,
        n=n,
        arr=arr,
        model=gower,
        data_type=data_type,
        verbose=verbose,
        backend=backend,
        matrix_type=matrix_type,
    )

    for i, row in results:
        if row.size > 0:
            result_matrix[i, i + 1 : n] = row

    result_matrix += result_matrix.T

    if matrix_type == "distance":
        np.fill_diagonal(result_matrix, 0.0)
    elif matrix_type == "similarity":  # pragma: no branch
        np.fill_diagonal(result_matrix, 1.0)
    else:
        msg = (
            f"Unknown matrix_type '{matrix_type}'. Must be 'distance' or 'similarity'."
        )
        raise ValueError(msg)

    if convert_to_sparse:
        return get_scipy_sparse_matrix(
            result_matrix,
            matrix_format=sparse_type,
            data_type=data_type,
        )

    return result_matrix


@contextmanager
def temporary_skip_oor(gower: "Gower"):
    """
    Temporarily disable out-of-range (OOR) checking on a Gower instance.

    This context manager sets `gower.skip_oor` to True for the duration of the
    context and restores its previous value afterward, even if an exception
    occurs.

    Parameters
    ----------
    gower (gower_metric.Gower): Fitted Gower instance.

    Yields
    ------
    None

    Notes
    -----
    This mutates shared state on the provided `gower` object. It is not
    thread-safe if the same instance is used concurrently in multiple
    computations.
    """
    old = gower.skip_oor
    gower.skip_oor = True
    try:
        yield
    finally:
        gower.skip_oor = old

def calculate_matrix(
    gower: "Gower",
    X: pd.DataFrame | np.ndarray,
    data_type: type[np.floating] | None = None,
    n_jobs: int = -1,
    verbose: int = 0,
    matrix_type: str = "distance",
    convert_to_sparse: bool = False,
    sparse_type: str = "csr",
    backend: str = "loky",
) -> (
    np.ndarray
    | scipy.sparse.csr_matrix
    | scipy.sparse.csc_matrix
    | scipy.sparse.coo_matrix
):
    """Return symmetric pairwise Gower distance matrix using joblib (parallel).

    Scipy sparse matrices are not supported as direct input.

    Args:
        gower (gower_metric.Gower): Fitted Gower instance.
        X (pd.DataFrame | np.ndarray): shape of (n_samples, n_features).
        data_type (type[np.floating] | None): data type used for the output distance matrix.
            If None, uses the data_type from the Gower instance configuration.
        n_jobs (int): number of parallel jobs to run, -1 means using all processors. Default is -1.
        verbose (int): whether to show tqdm progress bar. Default is 0 (no progress bar).
        matrix_type (str): Type of matrix to compute, either 'distance' or 'similarity'.
            Default is 'distance'.
        convert_to_sparse (bool): Whether to convert the output dense matrix to a sparse format.
            Default is False.
        sparse_type (str): Type of sparse matrix to convert to, either 'csr', 'csc' or 'coo'.
            Default is 'csr'.
        backend (str): Backend to use for joblib parallelization. Default is 'loky'.

    Returns:
        np.ndarray | scipy.sparse.csr_matrix | scipy.sparse.csc_matrix | scipy.sparse.coo_matrix:
            Pairwise Gower distance or similarity matrix of shape (n_samples, n_samples) or sparse matrix.

    Examples:
        Basic usage:
            >>> import pandas as pd
            >>> from gower_metric import Config, Gower
            >>> from gower_metric.utils.matrix.distance import calculate_matrix
            >>> data = pd.DataFrame({
            ...     'feature1': [1.0, 2.0, 3.0, 4.0],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            ...     'feature3': [0, 1, 0, 1],
            ...})
            >>> feature_types = {
            ...     'feature1': 'numeric',
            ...     'feature2': 'categorical_nominal',
            ...     'feature3': 'binary_symmetric',
            ... }
            >>> cfg = Config(
            ...     feature_types=feature_types,
            ... )
            >>> gower = Gower(cfg).fit(data)
            >>> similarity_matrix = calculate_matrix
            ...     gower,
            ...     data,
            ...     matrix_type='similarity',
            ...     convert_to_sparse=True,
            ...     sparse_type='csr'
            ... )

        Using similarity matrix and sparse output:
            >>> import pandas as pd
            >>> from gower_metric import Config, Gower
            >>> from gower_metric.utils.matrix.distance import calculate_matrix
            >>> data = pd.DataFrame({
            ...     'feature1': [1.0, 2.0, 3.0, 4.0],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            ...     'feature3': [0, 1, 0, 1],
            ... })
            >>> feature_types = {
            ...     'feature1': 'numeric',
            ...     'feature2': 'categorical_nominal',
            ...     'feature3': 'binary_symmetric',
            ... }
            >>> cfg = Config(
            ...     feature_types=feature_types,
            ... )
            >>> gower = Gower(cfg).fit(data)
            >>> similarity_matrix = calculate_matrix(
            ...     gower,
            ...     data,
            ...     matrix_type='similarity',
            ...     convert_to_sparse=True,
            ...     sparse_type='csr'
            ... )

    """
    if scipy.sparse.issparse(X):
        msg = "Sparse matrices are currently not supported as direct input. Please provide a dense matrix."
        raise ValueError(msg)

    if data_type is None:
        data_type = gower.data_type

    arr_check = (
        X.to_numpy(dtype=object)
        if isinstance(X, pd.DataFrame)
        else np.asarray(X, dtype=object)
    )

    if not gower.skip_oor:
        enforce_oor_policy(
            arr_check,
            strategy=gower.out_of_range,
            numeric_indices=gower.numeric_indices,
            numeric_mins=gower.numeric_mins,
            numeric_maxs=gower.numeric_maxs,
            ratio_scale_indices=gower.ratio_scale_indices,
            ratio_mins=gower.ratio_mins,
            ratio_maxs=gower.ratio_maxs,
            stacklevel=2,
        )

    with temporary_skip_oor(gower):
        return _get_full_matrix(
            gower,
            X,
            data_type=data_type,
            n_jobs=n_jobs,
            verbose=verbose,
            matrix_type=matrix_type,
            convert_to_sparse=convert_to_sparse,
            sparse_type=sparse_type,
            backend=backend,
        )