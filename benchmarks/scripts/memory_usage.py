import tracemalloc
import time
import numpy as np
import pandas as pd

from gower_metric import Gower
from sklearn.preprocessing import OneHotEncoder

from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed, cpu_count

def row_distances(i: int, df: pd.DataFrame, gower) -> np.ndarray:
    """
        Calculate distances for a single row against all other rows.
        """
    xi = df[i]
    return np.fromiter((gower(xi, df[j]) for j in range(len(df))), dtype=np.float16, count=len(df))

def row_upper(i: int, df: pd.DataFrame, gower) -> np.ndarray:
    """
    Calculate distances for a single row against all other rows, but only for the upper triangle of the matrix.
    """
    xi = df[i]
    return np.fromiter(
        (gower(xi, df[j]) if j > i else 0.0 for j in range(len(df))),
        dtype=np.float16,
        count=len(df),
    )


def run_gower_joblib(data):
    df = data.to_numpy()

    feature_types = {
        "age": "numeric",
        "education_num": "numeric",
        "sex": "categorical_nominal",
        "hours_per_week": "numeric",
    }

    gower = Gower(feature_types=feature_types).fit(df)

    matrix = np.zeros((len(df), len(df)), dtype=np.float16)

    with tqdm_joblib(desc="Calculating rows", total=len(df)) as _:
        rows = Parallel(n_jobs=cpu_count(), backend="multiprocessing", verbose=0)(
            delayed(row_distances)(i, df, gower) for i in range(len(df)))

    matrix = np.vstack(rows)

    return matrix

def run_gower_upper_joblib(data):
    df = data.to_numpy()

    feature_types = {
        "age": "numeric",
        "education_num": "numeric",
        "sex": "categorical_nominal",
        "hours_per_week": "numeric",
    }

    gower = Gower(feature_types=feature_types).fit(df)

    matrix = np.zeros((len(df), len(df)), dtype=np.float16)
    with tqdm_joblib(desc="Calculating upper rows", total=len(df)) as _:
        rows = Parallel(n_jobs=cpu_count(), backend="multiprocessing", verbose=0)(
            delayed(row_upper)(i, df, gower) for i in range(len(df)))
    
    matrix = np.vstack(rows)
    matrix += matrix.T

    np.fill_diagonal(matrix, 0)
    return matrix


def run_onehot(data):
    encoder = OneHotEncoder()
    encoded_df = encoder.fit_transform(data)
    return encoded_df

def main():
    data = pd.read_csv("./comparison/data/adult_reduced.csv").head(10000)

    print("=== Gower joblib===")
    tracemalloc.start()
    start = time.time()
    _ = run_gower_joblib(data)
    end = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Gower time (joblib): {end - start:.2f}s")
    print(f"Gower peak memory (joblib): {peak / 1024 / 1024:.2f} MB")

    print("\n=== Gower upper joblib ===")
    tracemalloc.start()
    start = time.time()
    _ = run_gower_upper_joblib(data)
    end = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Gower Upper time (joblib): {end - start:.2f}s")
    print(f"Gower Upper peak memory (joblib): {peak / 1024 / 1024:.2f} MB")

    print("\n=== One-Hot ===")
    tracemalloc.start()
    start = time.time()
    _ = run_onehot(data)
    end = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"OneHot time: {end - start:.2f}s")
    print(f"OneHot peak memory: {peak / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
