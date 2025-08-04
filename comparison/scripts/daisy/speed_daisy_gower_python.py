from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from tqdm.auto import tqdm

from gower_metric import Gower

df = pd.read_csv("your_path/adult_reduced.csv").head(1000)

feature_types: dict[int | str, str] = {
    "age": "ratio_scale_interval",
    "education_num": "ratio_scale_interval",
    "race": "categorical_nominal",
    "sex": "categorical_nominal",
    "hours_per_week": "ratio_scale_interval",
}

gower = Gower(feature_types, scale="range").fit(df)
arr = df.to_numpy(dtype=object)
n = arr.shape[0]


def single_distance_matrix() -> float:
    """
    Calculate the distance matrix for the dataset using Gower's distance.
    """

    def row_distances(i: int) -> np.ndarray:
        """
        Calculate distances for a single row against all other rows.
        """
        xi = arr[i]
        return np.fromiter(
            (gower(xi, arr[j]) for j in range(n)), dtype=np.float16, count=n
        )

    t0 = perf_counter()
    _ = Parallel(n_jobs=cpu_count(), backend="loky", verbose=0)(
        delayed(row_distances)(i) for i in range(n)
    )
    return perf_counter() - t0


times = []
for _ in tqdm(range(100), desc="100 experiments"):
    times.append(single_distance_matrix())
