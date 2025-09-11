import gc
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import seaborn as sns
from joblib import Parallel, cpu_count, delayed
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from gower_metric import Gower


def _load_data(dataset_id: int) -> pd.DataFrame:
    dataset = openml.datasets.get_dataset(dataset_id)
    X, _, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    X = X.drop(columns=["workclass", "occupation", "native-country"])
    return X


def row_upper(i: int, df: np.ndarray, gower) -> np.ndarray:
    """
    Calculate distances for a single row against all other rows, but only for the upper triangle of the matrix.
    """
    xi = df[i]
    return np.fromiter(
        (gower(xi, df[j]) if j > i else 0.0 for j in range(len(df))),
        dtype=np.float32,
        count=len(df),
    )


def gower_joblib_upper(
    DATASET_ID: int, n_rows: int, feature_types: dict
) -> tuple[np.ndarray, float]:
    gc.collect()
    data: pd.DataFrame = _load_data(DATASET_ID)
    data = data.head(n_rows)
    gower = Gower(feature_types=feature_types).fit(data)
    data = data.to_numpy()

    matrix = np.zeros((len(data), len(data)), dtype=np.float32)

    start_time = time.perf_counter()
    rows = Parallel(n_jobs=cpu_count(), backend="multiprocessing", verbose=0)(
        delayed(row_upper)(i, data, gower) for i in range(len(data))
    )

    matrix = np.vstack(rows)
    matrix += matrix.T
    np.fill_diagonal(matrix, 0)
    end_time = time.perf_counter()

    time_taken = end_time - start_time

    del data, gower, rows

    return matrix, time_taken


def gower_scipy(
    DATASET_ID: int, n_rows: int, feature_types: dict
) -> tuple[np.ndarray, float]:
    gc.collect()
    data: pd.DataFrame = _load_data(DATASET_ID)
    data = data.head(n_rows)
    gower = Gower(feature_types=feature_types).fit(data)

    start_time = time.perf_counter()
    dist_array = pdist(data, metric=gower)
    matrix_scipy = squareform(dist_array)
    end_time = time.perf_counter()

    time_taken = end_time - start_time

    del data, gower, dist_array

    return matrix_scipy, time_taken


def gower_sklearn(
    DATASET_ID: int, n_rows: int, feature_types: dict
) -> tuple[np.ndarray, float]:
    gc.collect()
    data: pd.DataFrame = _load_data(DATASET_ID)
    data = data.head(n_rows)
    gower = Gower(feature_types=feature_types).fit(data)

    start_time = time.perf_counter()
    matrix_sklearn = pairwise_distances(
        data, metric=gower, n_jobs=cpu_count(), ensure_all_finite=False
    )
    end_time = time.perf_counter()

    time_taken = end_time - start_time

    del data, gower

    return matrix_sklearn, time_taken


def _compare_matrices(
    mat1: np.ndarray, mat2: np.ndarray, mat3: np.ndarray, tolerance: float = 1e-5
) -> bool:
    if not np.allclose(mat1, mat2, atol=tolerance):
        sys.exit(1)

    if not np.allclose(mat1, mat3, atol=tolerance):
        sys.exit(1)

    if not np.allclose(mat2, mat3, atol=tolerance):
        sys.exit(1)


def plot_results(
    results: dict,
    max_range: int,
    do_show_plot: bool = False,
    do_save_plot: bool = False,
    resolution: int = 600,
) -> None:
    methods = list(results.keys())
    n_rows = list(results[methods[0]].keys())

    time_data = []
    for method in methods:
        for n in n_rows:
            mean_time, std_time = results[method][n]
            time_data.append(
                {
                    "Method": method,
                    "Number of Rows": n,
                    "Mean Time (s)": mean_time,
                    "Std Time (s)": std_time,
                }
            )

    time_df = pd.DataFrame(time_data)

    plt.figure(figsize=(14, 8))
    ax = sns.lineplot(
        data=time_df,
        x="Number of Rows",
        y="Mean Time (s)",
        hue="Method",
        marker="o",
        markersize=8,
    )

    # add std
    for method in methods:
        method_data = time_df[time_df["Method"] == method]
        ax.errorbar(
            method_data["Number of Rows"],
            method_data["Mean Time (s)"],
            yerr=method_data["Std Time (s)"],
            fmt="none",
            capsize=5,
            elinewidth=1,
            alpha=0.7,
        )

    plt.xticks(ticks=np.arange(100, max_range + 1, 100))
    plt.xlabel("Number of Rows")
    plt.ylabel("Execution Time (s)")
    plt.title(
        "Execution Time vs Number of Rows within Dataset (Number of Rows x 12 Features)"
    )
    plt.grid(True)
    plt.tight_layout()

    if do_save_plot:
        plt.savefig("gower_benchmark.png", dpi=resolution)

    if do_show_plot:
        plt.show()


def main() -> None:
    DATASET_ID: int = 1590
    results: dict = {}  # {method: {n_rows: (mean time [s], std time [s])}}
    all_times: dict = {}  # {method: {n_rows: [time1, time2, ...]}}

    feature_types = {
        "age": "numeric",
        "fnlwgt": "numeric",
        "education": "categorical_nominal",
        "education-num": "numeric",
        "marital-status": "categorical_nominal",
        "relationship": "categorical_nominal",
        "race": "categorical_nominal",
        "sex": "categorical_nominal",
        "capital-gain": "numeric",
        "capital-loss": "numeric",
        "hours-per-week": "numeric",
    }

    N_ITERATIONS = 5
    MAX_RANGE = 1000

    # time benchmark
    with tqdm(total=N_ITERATIONS * (MAX_RANGE // 100), desc="Overall Progress") as pbar:
        for curr_iter in range(N_ITERATIONS):
            for n_rows in range(100, MAX_RANGE + 1, 100):
                pbar.set_postfix({"Iteration": f"{curr_iter + 1} | {N_ITERATIONS}"})
                pbar.update(1)

                gower_joblib_upper_matrix, joblib_time = gower_joblib_upper(
                    DATASET_ID, n_rows, feature_types
                )
                all_times.setdefault("joblib (multiprocessing)", {}).setdefault(
                    n_rows, []
                ).append(joblib_time)

                gower_scipy_matrix, scipy_time = gower_scipy(
                    DATASET_ID, n_rows, feature_types
                )
                all_times.setdefault("scipy", {}).setdefault(n_rows, []).append(
                    scipy_time
                )

                gower_sklearn_matrix, sklearn_time = gower_sklearn(
                    DATASET_ID, n_rows, feature_types
                )
                all_times.setdefault("sklearn", {}).setdefault(n_rows, []).append(
                    sklearn_time
                )

                _compare_matrices(
                    gower_joblib_upper_matrix, gower_scipy_matrix, gower_sklearn_matrix
                )

                del gower_sklearn_matrix, sklearn_time

    for method, times_dict in all_times.items():
        for n_rows, times in times_dict.items():
            mean_time = np.mean(times)
            std_time = np.std(times)
            results.setdefault(method, {})[n_rows] = (mean_time, std_time)

    plot_results(results, MAX_RANGE, do_save_plot=True, resolution=2400)


if __name__ == "__main__":
    main()
