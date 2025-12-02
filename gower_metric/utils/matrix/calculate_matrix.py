from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse
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
    data_type: type[np.floating | np.integer],
    row_type: str,
) -> tuple[int, np.ndarray]:
    """Compute one upper triangle row of Gower distances.

    Args:
        i (int): row index.
        X_arr (np.ndarray): data array of shape (n_samples, n_features).
        n (int): number of samples.
        model (Gower): fitted Gower instance.
        data_type (type[np.integer | np.floating]): data type for the output row array.
        row_type (str): type of row to compute, distance or similarity. Defaults to "distance".

    Returns:
        tuple[i, row]: tuple of row index and computed row array.
    """
    n = X_arr.shape[0] if n == 0 else n
    xi = X_arr[i]

    start = i + 1
    count = n - start
    row = np.zeros(n, dtype=data_type)

    if count <= 0:
        return (i, row)

    func = model.similarity if row_type == "similarity" else model

    values = (func(xi, X_arr[j]) for j in range(start, n))

    row[start:n] = np.fromiter(values, dtype=data_type, count=count)

    return (i, row)


def _get_results_from_joblib(
    arr: np.ndarray,
    n_jobs: int,
    verbose: int,
    data_type: type[np.floating | np.integer],
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
        data_type (type[np.floating | np.integer]): data type for the output rows.
        model (Gower): fitted Gower instance.
        matrix_type (str): type of matrix to compute, distance or similarity. Defaults to "distance".
        backend (str): joblib backend to use. Defaults to "loky".
        n (int): number of samples (if 0, will be set to arr.shape[0]).

    Returns:
        list[tuple[int, np.ndarray]]: List of tuples (row index, computed row array).
    """
    results: list[tuple[int, np.ndarray]] = Parallel(
        n_jobs=n_jobs, backend=backend, verbose=verbose
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


def get_full_matrix(
    self: "Gower",
    X: pd.DataFrame | np.ndarray,
    data_type: type[np.floating | np.integer] = np.float32,
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
        self (Gower): Fitted Gower instance.
        X (pd.DataFrame | np.ndarray): shape of (n_samples, n_features).
        data_type (type[np.floating | np.integer]): data type for the output distance matrix, default np.float32.
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
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=object)
    else:
        arr = np.array(X, dtype=object)

    n: int = arr.shape[0]

    MATRIX: np.ndarray = np.zeros((n, n), dtype=data_type)

    results: list[tuple[int, np.ndarray]] = _get_results_from_joblib(
        n_jobs=n_jobs,
        n=n,
        arr=arr,
        model=self,
        data_type=data_type,
        verbose=verbose,
        backend=backend,
        matrix_type=matrix_type,
    )

    for i, row in results:
        MATRIX[i] = row

    MATRIX += MATRIX.T

    if matrix_type == "distance":
        np.fill_diagonal(MATRIX, 0.0)
    elif matrix_type == "similarity":
        np.fill_diagonal(MATRIX, 1.0)

    if convert_to_sparse:
        MATRIX = get_scipy_sparse_matrix(
            MATRIX, matrix_format=sparse_type, data_type=data_type
        )

    return MATRIX
