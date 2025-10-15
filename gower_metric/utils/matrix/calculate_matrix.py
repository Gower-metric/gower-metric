from typing import TYPE_CHECKING

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

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
        i: Row index.
        X_arr: Data array (n_samples, n_features).
        n: Number of samples.
        model: Fitted Gower instance.
        data_type: Data type for the output row array.
        row_type: Type of row to compute, distance or similarity. Defaults to "distance".

    Returns:
        (i, row): Tuple of row index and computed row array.
    """
    n = X_arr.shape[0] if n == 0 else n
    xi = X_arr[i]
    row = np.zeros(n, dtype=data_type)

    for j in range(i + 1, n):
        if row_type == "distance":
            row[j] = model(xi, X_arr[j])
        else:
            row[j] = model.similarity(xi, X_arr[j])

    return (i, row)


def get_results_from_joblib(
    arr: np.ndarray,
    n_jobs: int,
    verbose: int,
    data_type: type[np.floating | np.integer],
    model: "Gower",
    matrix_type: str,
    n: int = 0,
    backend: str = "multiprocessing",
) -> list[tuple[int, np.ndarray]]:
    """Get results from joblib parallel processing.

    Args:
        arr: Data array (n_samples, n_features).
        n_jobs: Number of parallel jobs.
        verbose: Whether to show progress bar.
        data_type: Data type for the output rows.
        model: Fitted Gower instance.
        matrix_type: Type of matrix to compute, distance or similarity. Defaults to "distance".
        n: Number of samples (if 0, will be set to arr.shape[0]).
        backend: Joblib backend to use. Defaults to "multiprocessing".

    Returns:
        List of tuples (row index, computed row array).
    """
    results: list[tuple[int, np.ndarray]] = Parallel(
        n_jobs=n_jobs, backend=backend, verbose=0
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
