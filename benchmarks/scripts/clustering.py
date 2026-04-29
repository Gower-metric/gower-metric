import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from openml.tasks import TaskType
from sklearn.cluster import HDBSCAN
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from gower_metric import Config, Gower

warnings.simplefilter(action="ignore", category=FutureWarning)


def _load_data(dataset_id: int, n_rows: int = 5_000) -> pd.DataFrame:
    dataset = openml.datasets.get_dataset(dataset_id)
    X, _, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe",
    )

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
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


def _compute_gower_row(i, X: np.ndarray, gower: Gower):
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


def _print_results(df: pd.DataFrame) -> None:
    df_clean = df.dropna(subset=["OHE", "Gower"])
    valid_count = len(df_clean)

    df_long = df_clean.melt(
        id_vars=["task_id"],
        value_vars=["OHE", "Gower"],
        var_name="Method",
        value_name="Score",
    )

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=df_long, x="Method", y="Score", ax=ax1, palette="Set2")
    sns.stripplot(data=df_long, x="Method", y="Score", ax=ax1, color=".3", alpha=0.5)
    ax1.set_title(f"Distribution of Scores (n={valid_count})")

    sns.scatterplot(data=df_clean, x="OHE", y="Gower", ax=ax2, s=100, alpha=0.7)

    [
        max(ax2.get_xlim()[0], ax2.get_ylim()[0]),
        min(ax2.get_xlim()[1], ax2.get_ylim()[1]),
    ]
    ax2.plot(
        [-1, 1],
        [-1, 1],
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Identity Line",
    )

    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_title("Direct Comparison per Dataset")
    ax2.set_xlabel("Silhouette Score (OneHotEncoding)")
    ax2.set_ylabel("Silhouette Score (Gower)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Clustering Benchmark: Gower Distance vs OneHotEncoding\nValid paired results: {valid_count}/{len(df)}",
        fontsize=16,
    )

    output_dir = Path(__file__).parent.parent.parent / "data" / "imgs" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "clustering_results.svg"

    plt.savefig(output_path)
    plt.close()


def main() -> None:
    clustering_tasks = openml.tasks.list_tasks(
        task_type=TaskType.CLUSTERING,
        output_format="dataframe",
    )
    task_ids = clustering_tasks["tid"].astype(int).tolist()

    n_tasks_to_run = 100
    all_results = []

    for task_id in tqdm(task_ids[:n_tasks_to_run], desc="Benchmarking Tasks"):
        try:
            task = openml.tasks.get_task(task_id)
            X_raw = _load_data(task.dataset_id)
            X = _impute_data(X_raw)

            # OneHotEncoding
            enc_data = _encode_data(X.copy())
            clusterer_ohe = HDBSCAN(
                metric="euclidean",
                min_cluster_size=10,
                min_samples=5,
            )
            clusterer_ohe.fit(enc_data)

            ohe_score = None
            if clusterer_ohe.labels_.max() > 0:
                ohe_score = silhouette_score(enc_data, clusterer_ohe.labels_)

            # Gower
            gower_features = _get_gower_features(X)
            cfg = Config(feature_types=gower_features)
            gower_engine = Gower(cfg).fit(X)
            gower_matrix = _get_gower_matrix(X, gower_engine)

            clusterer_gower = HDBSCAN(
                metric="precomputed",
                min_cluster_size=10,
                min_samples=5,
            )
            clusterer_gower.fit(gower_matrix)

            gower_score = None
            if clusterer_gower.labels_.max() > 0:
                gower_score = silhouette_score(
                    gower_matrix,
                    clusterer_gower.labels_,
                    metric="precomputed",
                )

            all_results.append(
                {
                    "task_id": task_id,
                    "OHE": ohe_score,
                    "Gower": gower_score,
                },
            )

        except Exception:
            pass

    df_results = pd.DataFrame(all_results)
    _print_results(df_results)


if __name__ == "__main__":
    main()
