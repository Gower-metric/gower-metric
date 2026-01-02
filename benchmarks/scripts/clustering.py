import warnings

import numpy as np
import openml
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from openml.tasks import TaskType
from sklearn.cluster import HDBSCAN
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from gower_metric import Gower
from gower_metric.core.config import Config

warnings.simplefilter(action="ignore", category=FutureWarning)


def _load_data(dataset_id: int, n_rows: int = 5_000) -> pd.DataFrame:
    dataset = openml.datasets.get_dataset(dataset_id)
    X, _, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )

    return X.head(n_rows)


def _impute_data(X: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    if len(numeric_cols) > 0:
        num_imputer = KNNImputer(n_neighbors=5)
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy="constant")
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    return X


def _encode_data(X: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_enc = encoder.fit_transform(X[categorical_cols])

        X = X.drop(columns=categorical_cols)

        X = pd.concat([X.reset_index(drop=True), pd.DataFrame(X_enc)], axis=1)
    else:
        X = X.reset_index(drop=True)

    X.columns = X.columns.astype(str)

    return X


def _get_gower_features(X: pd.DataFrame) -> dict:
    gower_features = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            gower_features[col] = "numeric"
        else:
            gower_features[col] = "categorical_nominal"
    return gower_features


def _compute_gower_row(i, X: np.asarray, gower: Gower):
    xi = X[i]
    return np.fromiter(
        (gower(xi, X[j]) if j > i else 0.0 for j in range(X.shape[0])),
        dtype=np.float32,
        count=X.shape[0],
    )


def _get_gower_matrix(X: pd.DataFrame, gower: Gower) -> np.ndarray:
    X = X.to_numpy()

    matrix = np.zeros((len(X), len(X)), dtype=np.float32)

    _matrix_rows = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(_compute_gower_row)(i, X, gower) for i in range(X.shape[0])
    )
    matrix = np.vstack(_matrix_rows)
    matrix += matrix.T
    np.fill_diagonal(matrix, 0.0)

    return matrix


def _print_results(
    results_enc: list[float], results_gower: list[float], valid_results: int
) -> print:
    print(
        f"Average silhouette score over {valid_results} tasks (OneHotEncoding): {sum(results_enc) / valid_results:.4f}"
    )
    print(f"Standard deviation of silhouette score: {pd.Series(results_enc).std():.4f}")

    print(
        f"Average silhouette score over {valid_results} tasks (Gower): {sum(results_gower) / valid_results:.4f}"
    )
    print(
        f"Standard deviation of silhouette score: {pd.Series(results_gower).std():.4f}"
    )


def main():
    clustering_tasks = openml.tasks.list_tasks(
        task_type=TaskType.CLUSTERING, output_format="dataframe"
    )
    task_ids = clustering_tasks["tid"].astype(int).tolist()

    strategies = ["onehotencoding", "gower"]

    results_enc = []
    results_gower = []
    valid_results: int = 0

    n_tasks_to_run = 100

    with tqdm(total=len(strategies) * n_tasks_to_run) as pbar:
        for strategy in strategies:
            for task_id in task_ids[:n_tasks_to_run]:
                pbar.set_description(f"Strategy: {strategy}, Task ID: {task_id}")
                pbar.update(1)

                task = openml.tasks.get_task(task_id)

                X = _load_data(task.dataset_id)
                X = _impute_data(X)

                if strategy == "gower":
                    gower_features = _get_gower_features(X)

                    if type(gower_features) is not dict:
                        raise ValueError("gower_features must be a dictionary")
                    
                    cfg = Config(
                        feature_types=gower_features
                    )
                    gower = Gower(cfg).fit(X)

                if strategy == "onehotencoding":
                    enc_data = _encode_data(X)

                    with parallel_backend("multiprocessing"):
                        clusterer = HDBSCAN(
                            metric="euclidean",
                            min_cluster_size=10,
                            min_samples=5,
                            n_jobs=-1,
                        )
                        clusterer.fit(enc_data)

                    if clusterer.labels_.max() > 1:
                        results_enc.append(
                            silhouette_score(enc_data, clusterer.labels_)
                        )
                        valid_results += 1

                if strategy == "gower":
                    gower_matrix = _get_gower_matrix(X, gower)

                    with parallel_backend("multiprocessing"):
                        clusterer = HDBSCAN(
                            metric="precomputed",
                            min_cluster_size=10,
                            min_samples=5,
                            n_jobs=-1,
                        )
                        clusterer.fit(gower_matrix)

                    # make sure that there are more than 1 cluster
                    if clusterer.labels_.max() > 1:
                        results_gower.append(
                            silhouette_score(
                                gower_matrix, clusterer.labels_, metric="precomputed"
                            )
                        )
                        valid_results += 1

    _print_results(results_enc, results_gower, valid_results)


if __name__ == "__main__":
    main()
