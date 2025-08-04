"""
Here we provide quick how-to speed up matrix calculation using Gower's similarity method.
There are few approaches to speed up the computation:
- Using `joblib` to parallelize the computation across multiple CPU cores.
- calculate half of the matrix and then mirror it to get the full matrix.

For now, there is no numba or cython implementation, but it is planned for the future.
Feel free to use this script as a speedup reference for your own projects.
You can also compare the results with R environment.
"""

from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from tqdm.auto import tqdm

from gower_metric import Gower

df = pd.read_csv("comparison/data/adult_reduced.csv").head(1000)

feature_types: dict[int | str, str] = {
    "age": "ratio_scale_interval",
    "education_num": "ratio_scale_interval",
    "race": "categorical_nominal",
    "sex": "categorical_nominal",
    "hours_per_week": "ratio_scale_interval",
}

gower = Gower(feature_types, scale="range").fit(df)
n = len(df)
arr = df.to_numpy(dtype=object)


def row_distances(i: int) -> np.ndarray:
    """
    Calculate distances for a single row against all other rows.
    """
    xi = arr[i]
    return np.fromiter((gower(xi, arr[j]) for j in range(n)), dtype=np.float16, count=n)


def row_upper(i: int):
    """
    Calculate distances for a single row against all other rows, but only for the upper triangle of the matrix.
    """
    xi = arr[i]
    return np.fromiter(
        (gower(xi, arr[j]) if j > i else 0.0 for j in range(n)),
        dtype=np.float16,
        count=n,
    )


# matrix with no speedup
matrix = np.zeros((n, n), dtype=np.float16)
print("Calculating distance matrix without speedup...")
t0 = perf_counter()
for i in tqdm(range(n), desc="loop", unit="row"):
    xi = arr[i]
    for j in range(n):
        matrix[i, j] = gower(xi, arr[j])
t1 = perf_counter()
print(f"Distance matrix calculation without speedup took {t1 - t0:.2f} seconds.")

# matrix with speedup using joblib
matrix_joblib = np.zeros((n, n), dtype=np.float16)
print("Calculating distance matrix using parallel processing...")
t2 = perf_counter()
rows = Parallel(n_jobs=cpu_count(), backend="loky", verbose=0)(
    delayed(row_distances)(i)
    for i in tqdm(range(n), desc="Calculating rows", unit="row")
)
matrix_joblib = np.vstack(rows)
t3 = perf_counter()
print(f"Distance matrix calculation with speedup took {t3 - t2:.2f} seconds.")

# matrix with speedup using joblib and upper triangle
matrix_upper = np.zeros((n, n), dtype=np.float16)
print("Calculating distance matrix using parallel processing for upper triangle...")
t4 = perf_counter()
rows_upper = Parallel(n_jobs=cpu_count(), backend="loky", verbose=0)(
    delayed(row_upper)(i)
    for i in tqdm(range(n), desc="Calculating upper triangle rows", unit="row")
)
matrix_upper = np.vstack(rows_upper)
matrix_upper += matrix_upper.T
np.fill_diagonal(matrix_upper, 0.0)
t5 = perf_counter()
print(
    f"Distance matrix calculation with upper triangle speedup took {t5 - t4:.2f} seconds."
)


def compare_matrices(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    matrix3: np.ndarray,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compare the three matrices to ensure they are equal.
    """
    return (
        np.allclose(matrix1, matrix2, atol=tolerance)
        and np.allclose(matrix1, matrix3, atol=tolerance)
        and np.allclose(matrix2, matrix3, atol=tolerance)
    )


compare_matrices(matrix, matrix_joblib, matrix_upper)
print("All matrices are equal within the specified tolerance.")


def compare_with_R(
    matrix_upper: np.ndarray, r_matrix: np.ndarray, tolerance: float = 1e-5
) -> bool:
    """
    Compare the upper triangle matrix with the R matrix to ensure they are equal.
    """
    return np.allclose(matrix_upper, r_matrix, atol=tolerance)


matrix_R = np.loadtxt(fname="path/file.txt")
compare_with_R(matrix_upper, matrix_R)
print("Upper triangle matrix is equal to the R matrix within the specified tolerance.")
