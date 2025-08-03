import numpy as np
import pandas as pd

from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm

from gower_metric import Gower

df = pd.read_csv("comparison/data/adult_reduced.csv").head(1000)

feature_types = {
    "age": "ratio_scale_interval",
    "education_num": "ratio_scale_interval",
    "race": "categorical_nominal",
    "sex": "categorical_nominal",
    "hours_per_week": "ratio_scale_interval",
}

feature_weights = {
    0: 1.0,
    1: 2.0,
    2: 3.0,
    3: 4.0,
    4: 5.0,
}

gower = Gower(feature_types, feature_weights=feature_weights).fit(df)
n = len(df)
arr = df.to_numpy(dtype=object)

def row_upper(i: int):
    """
    Calculate distances for a single row against all other rows, but only for the upper triangle of the matrix.
    """
    xi = arr[i]
    return np.fromiter((gower(xi, arr[j]) if j > i else 0.0 for j in range(n)), dtype=np.float16, count=n)

matrix_upper = np.zeros((n, n), dtype=np.float16)
rows_upper = Parallel(n_jobs=cpu_count(), backend="loky", verbose=0)(
    delayed(row_upper)(i) for i in tqdm(range(n), desc="Calculating upper triangle rows", unit="row")
)
matrix_upper = np.vstack(rows_upper)
matrix_upper += matrix_upper.T