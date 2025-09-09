import tracemalloc
import time
import numpy as np
import pandas as pd

from gower_metric import Gower
from sklearn.preprocessing import OneHotEncoder

from tqdm.auto import tqdm
from joblib import Parallel, delayed, cpu_count

def row_distances(i: int, df: pd.DataFrame, gower) -> np.ndarray:
    """
        Calculate distances for a single row against all other rows.
        """
    xi = df[i]
    return np.fromiter((gower(xi, df[j]) for j in range(len(df))), dtype=np.float32, count=len(df))


def run_gower_joblib(data):
    df = data.to_numpy()

    feature_types = {
        "age": "numeric",
        "education_num": "numeric",
        "sex": "categorical_nominal",
        "hours_per_week": "numeric",
    }

    gower = Gower(feature_types=feature_types).fit(df)

    matrix = np.zeros((len(df), len(df)), dtype=np.float32)
    rows = Parallel(n_jobs=cpu_count(), backend="multiprocessing", verbose=0)(
        delayed(row_distances)(i, df, gower)
        for i in tqdm(range(len(df)), desc="Calculating rows", unit="row"))
    matrix = np.vstack(rows)

    return matrix

def run_onehot(data):
    encoder = OneHotEncoder(sparse_output=False)
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
