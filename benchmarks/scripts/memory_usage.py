import gc
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import seaborn as sns
from joblib import Parallel, cpu_count, delayed
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from gower_metric import Gower
from gower_metric.core.config import Config


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
    cfg = Config(
        feature_types=feature_types
    )
    gower = Gower(cfg).fit(data)
    data = data.to_numpy()

    matrix = np.zeros((len(data), len(data)), dtype=np.float32)

    tracemalloc.start()
    rows = Parallel(n_jobs=cpu_count(), backend="multiprocessing", verbose=0)(
        delayed(row_upper)(i, data, gower) for i in range(len(data))
    )

    matrix = np.vstack(rows)
    matrix += matrix.T
    np.fill_diagonal(matrix, 0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    del data, gower, rows

    return matrix, peak


def gower_onehot_enc(
    DATASET_ID: int, n_rows: int, feature_types: dict
) -> tuple[np.ndarray, float]:
    gc.collect()
    data: pd.DataFrame = _load_data(DATASET_ID)
    data = data.head(n_rows)
    cfg = Config(
        feature_types=feature_types
    )
    gower = Gower(cfg).fit(data)
    data = data.to_numpy()

    tracemalloc.start()
    enc = OneHotEncoder(sparse_output=False)
    data_enc = enc.fit_transform(data)
    matrix = distance_matrix(data_enc, data_enc)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    del data, gower, data_enc

    return matrix, peak


def gower_scipy(
    DATASET_ID: int, n_rows: int, feature_types: dict
) -> tuple[np.ndarray, float]:
    gc.collect()
    data: pd.DataFrame = _load_data(DATASET_ID)
    data = data.head(n_rows)
    cfg = Config(
        feature_types=feature_types
    )
    gower = Gower(cfg).fit(data)

    tracemalloc.start()
    dist_array = pdist(data, metric=gower)
    matrix_scipy = squareform(dist_array)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    del data, gower, dist_array

    return matrix_scipy, peak


def gower_sklearn(
    DATASET_ID: int, n_rows: int, feature_types: dict
) -> tuple[np.ndarray, float]:
    gc.collect()
    data: pd.DataFrame = _load_data(DATASET_ID)
    data = data.head(n_rows)
    cfg = Config(
        feature_types=feature_types
    )
    gower = Gower(cfg).fit(data)

    tracemalloc.start()
    matrix_sklearn = pairwise_distances(
        data, metric=gower, n_jobs=cpu_count(), ensure_all_finite=False
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    del data, gower

    return matrix_sklearn, peak


def plot_results(
    results: dict,
    max_range: int,
    iterations: int,
    do_show_plot: bool = False,
    do_save_plot: bool = False,
    resolution: int = 600,
) -> None:
    methods = list(results.keys())
    n_rows = list(results[methods[0]].keys())

    mem_data = []
    for method in methods:
        for n in n_rows:
            mean_mem, std_mem = results[method][n]
            mem_data.append(
                {
                    "Method": method,
                    "Number of Rows": n,
                    "Mean Memory (MB)": mean_mem,
                    "Std Memory (MB)": std_mem,
                }
            )

    mem_df = pd.DataFrame(mem_data)

    plt.figure(figsize=(14, 8))
    ax = sns.lineplot(
        data=mem_df,
        x="Number of Rows",
        y="Mean Memory (MB)",
        hue="Method",
        marker="o",
        markersize=8,
    )

    for method in methods:
        method_data = mem_df[mem_df["Method"] == method]
        ax.errorbar(
            method_data["Number of Rows"],
            method_data["Mean Memory (MB)"],
            yerr=method_data["Std Memory (MB)"],
            fmt="none",
            capsize=5,
            elinewidth=1,
            alpha=0.7,
        )

    plt.xticks(ticks=np.arange(100, max_range + 1, 100))
    plt.xlabel("Number of Rows")
    plt.ylabel("Peak Memory Usage (MB)")
    plt.title(
        "Peak Memory Usage calculating Gower matrix using different methods \
            and OneHot encoding Euclidean distance matrix (Number of rows x 12 features)"
    )
    plt.grid(True)
    plt.tight_layout()

    if do_save_plot:
        plt.savefig(
            f"benchmarks/imgs/memory_usage/benchmark_{iterations}_iterations.png",
            dpi=resolution,
        )

    if do_show_plot:
        plt.show()


def main():
    DATASET_ID: int = 1590
    results: dict = {}  # {method: {n_rows: (peak memory [bytes], std memory [bytes])}}
    all_memory: dict = {}  # {method: {n_rows: [peak memory [bytes], ...]}}

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

    N_ITERATIONS = 10
    MAX_RANGE = 1000

    with tqdm(total=N_ITERATIONS * (MAX_RANGE // 100), desc="Overall Progress") as pbar:
        for curr_iter in range(N_ITERATIONS):
            for n_rows in range(100, MAX_RANGE + 1, 100):
                pbar.set_postfix({"Iteration": f"{curr_iter + 1} | {N_ITERATIONS}"})
                pbar.update(1)

                gower_joblib, mem_gower_joblib = gower_joblib_upper(
                    DATASET_ID, n_rows, feature_types
                )
                all_memory.setdefault(
                    "Custom Gower (joblib multiprocessing)", {}
                ).setdefault(n_rows, []).append(mem_gower_joblib)

                gower_onehot, mem_gower_onehot = gower_onehot_enc(
                    DATASET_ID, n_rows, feature_types
                )
                all_memory.setdefault("OneHot encoding (Euclidean)", {}).setdefault(
                    n_rows, []
                ).append(mem_gower_onehot)

                gower_scipy_m, mem_gower_scipy = gower_scipy(
                    DATASET_ID, n_rows, feature_types
                )
                all_memory.setdefault("Gower scipy", {}).setdefault(n_rows, []).append(
                    mem_gower_scipy
                )

                gower_sklearn_m, mem_gower_sklearn = gower_sklearn(
                    DATASET_ID, n_rows, feature_types
                )
                all_memory.setdefault("Gower sklearn", {}).setdefault(
                    n_rows, []
                ).append(mem_gower_sklearn)

                del (
                    gower_joblib,
                    gower_onehot,
                    mem_gower_joblib,
                    mem_gower_onehot,
                    gower_scipy_m,
                    mem_gower_scipy,
                    gower_sklearn_m,
                    mem_gower_sklearn,
                )

    for method, n_rows_dict in all_memory.items():
        for n_rows, mem_values in n_rows_dict.items():
            mean_mem = np.mean(mem_values) / (1024**2)
            std_mem = np.std(mem_values) / (1024**2)
            results.setdefault(method, {})[n_rows] = (mean_mem, std_mem)

    plot_results(
        results,
        MAX_RANGE,
        N_ITERATIONS,
        resolution=2400,
    )


if __name__ == "__main__":
    main()
