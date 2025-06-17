import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm
from time import perf_counter

from gower_similarity.core.similarity import GowerSimilarity

df = pd.read_csv("comparison/data/adult_reduced.csv").head(1000)

feature_types = {
    "age": "ratio_scale_interval",
    "education_num": "ratio_scale_interval",
    "race": "categorical_nominal",
    "sex": "categorical_nominal",
    "hours_per_week": "ratio_scale_interval",
}

gs = GowerSimilarity(feature_types, scale="range").fit(df)
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
        return np.fromiter((gs.distance(xi, arr[j]) for j in range(n)), dtype=np.float16, count=n)

    t0 = perf_counter()
    _ = Parallel(n_jobs=cpu_count(), backend="loky", verbose=0)(
            delayed(row_distances)(i) for i in range(n)
        )
    return perf_counter() - t0

times = []
for _ in tqdm(range(100), desc="100 experiments"):
    times.append(single_distance_matrix())