import warnings

import numpy as np
import openml
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from gower_metric import Config, Gower

warnings.simplefilter(action="ignore", category=FutureWarning)


def _load_data(dataset_id: int, n_rows: int = 5_000) -> tuple[pd.DataFrame, pd.Series]:
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe",
    )
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    return X.head(n_rows), y.head(n_rows)


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def _impute_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

    if len(numeric_cols) > 0:
        num_imputer = KNNImputer(n_neighbors=5)
        X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    return X_train, X_test


def _encode_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])

        X_train = X_train.drop(columns=categorical_cols)
        X_test = X_test.drop(columns=categorical_cols)

        X_train = pd.concat(
            [X_train.reset_index(drop=True), pd.DataFrame(X_train_encoded)],
            axis=1,
        )
        X_test = pd.concat(
            [X_test.reset_index(drop=True), pd.DataFrame(X_test_encoded)],
            axis=1,
        )
    else:
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

    return X_train, X_test


def _grid_search_knn(
    y_train: pd.Series,
    X_train: pd.DataFrame = None,
    metric=None,
    train_matrix=None,
    backend: str = "loky",
) -> KNeighborsClassifier:
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
        "leaf_size": [20, 30, 40],
    }

    if metric == "precomputed":
        knn = KNeighborsClassifier(n_jobs=-1, metric="precomputed")
        grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, scoring="accuracy")

        with parallel_backend(backend):
            grid_search.fit(train_matrix, y_train)

    else:
        knn = KNeighborsClassifier(n_jobs=-1)
        grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, scoring="accuracy")

        with parallel_backend(backend):
            grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


def _get_gower_features(X: pd.DataFrame) -> dict:
    gower_features = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            gower_features[col] = "numeric"
        else:
            gower_features[col] = "categorical_nominal"
    return gower_features


def compute_gower_train(i, X_train_np, gower):
    xi = X_train_np[i]
    return np.fromiter(
        (
            gower(xi, X_train_np[j]) if j > i else 0.0
            for j in range(X_train_np.shape[0])
        ),
        dtype=np.float32,
        count=X_train_np.shape[0],
    )


def _get_gower_matrix_train(
    train_matrix: np.ndarray,
    gower: Gower,
    X_train: pd.DataFrame,
    backend: str = "loky",
) -> np.ndarray:
    X_train_np = X_train.to_numpy()

    train_matrix_rows = Parallel(n_jobs=-1, backend=backend)(
        delayed(compute_gower_train)(i, X_train_np, gower)
        for i in range(X_train_np.shape[0])
    )
    train_matrix = np.vstack(train_matrix_rows)
    train_matrix += train_matrix.T
    np.fill_diagonal(train_matrix, 0.0)

    return train_matrix


def compute_gower_test(i, X_test_np, X_train_np, gower):
    xi = X_test_np[i]
    return np.fromiter(
        (gower(xi, X_train_np[j]) for j in range(X_train_np.shape[0])),
        dtype=np.float32,
        count=X_train_np.shape[0],
    )


def _get_gower_matrix_test(
    test_matrix: np.ndarray,
    gower: Gower,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    backend: str = "loky",
) -> np.ndarray:
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()

    test_matrix = Parallel(n_jobs=-1, backend=backend)(
        delayed(compute_gower_test)(i, X_test_np, X_train_np, gower)
        for i in range(X_test_np.shape[0])
    )
    return np.array(test_matrix)


def main() -> None:
    suite = openml.study.get_suite(99)
    tasks = suite.tasks

    N_ROWS_TO_USE = 100_000
    joblib_backend = "loky"

    strategies = ["onehotencoding", "gower"]
    results_enc = []
    results_f1_enc = []
    results_gower = []
    results_f1_gower = []
    classification_report_enc: list[dict] = []
    classification_report_gower: list[dict] = []

    results_df: pd.DataFrame = pd.DataFrame()

    tasks = suite.tasks
    if tasks is None:
        msg = "No tasks found in suite"
        raise ValueError(msg)
    n_tasks_to_run = len(tasks)

    with tqdm(total=len(strategies) * n_tasks_to_run) as pbar:
        for strategy in strategies:
            for task_id in tasks[:n_tasks_to_run]:
                pbar.set_description(f"Strategy: {strategy}, Task ID: {task_id}")
                pbar.update(1)

                task = openml.tasks.get_task(task_id)

                X, y = _load_data(task.dataset_id, n_rows=N_ROWS_TO_USE)
                n_rows_used: int = X.shape[0]

                X_train, X_test, y_train, y_test = _split_data(X, y)
                X_train, X_test = _impute_data(X_train, X_test)

                if strategy == "gower":
                    gower_features = _get_gower_features(X_train)

                    if type(gower_features) is not dict:
                        msg = "gower_features must be a dictionary"
                        raise ValueError(msg)

                    cfg = Config(
                        feature_types=gower_features,
                    )
                    gower = Gower(cfg).fit(X_train)

                if strategy == "onehotencoding":
                    X_train, X_test = _encode_data(X_train, X_test)

                    X_train.columns = X_train.columns.astype(str)
                    X_test.columns = X_test.columns.astype(str)

                if strategy == "gower":
                    train_matrix = np.zeros(
                        (X_train.shape[0], X_train.shape[0]),
                        dtype=np.float32,
                    )
                    test_matrix = np.zeros(
                        (X_test.shape[0], X_train.shape[0]),
                        dtype=np.float32,
                    )

                    train_matrix = _get_gower_matrix_train(
                        train_matrix,
                        gower,
                        X_train,
                        joblib_backend,
                    )
                    test_matrix = _get_gower_matrix_test(
                        test_matrix,
                        gower,
                        X_train,
                        X_test,
                        joblib_backend,
                    )

                    knn = _grid_search_knn(
                        y_train=y_train,
                        metric="precomputed",
                        train_matrix=train_matrix,
                        backend=joblib_backend,
                    )

                    prefictions = knn.predict(test_matrix)
                    accuracy = accuracy_score(y_test, prefictions)
                    results_gower.append(accuracy)

                    f1 = f1_score(y_test, prefictions, average="weighted")
                    results_f1_gower.append(f1)

                    classification_report_gower.append(
                        classification_report(y_test, prefictions, output_dict=True),
                    )

                    results_df = pd.concat(
                        [
                            results_df,
                            pd.DataFrame(
                                [
                                    {
                                        "task_id": task_id,
                                        "strategy": strategy,
                                        "accuracy": accuracy,
                                        "f1": f1,
                                        "dataset_id": task.dataset_id,
                                        "n_rows": n_rows_used,
                                    },
                                ],
                            ),
                        ],
                        ignore_index=True,
                    )
                else:
                    knn = _grid_search_knn(
                        y_train=y_train,
                        X_train=X_train,
                        backend=joblib_backend,
                    )

                    prefictions = knn.predict(X_test)
                    accuracy = accuracy_score(y_test, prefictions)
                    results_enc.append(accuracy)

                    f1 = f1_score(y_test, prefictions, average="weighted")
                    results_f1_enc.append(f1)

                    classification_report_enc.append(
                        classification_report(y_test, prefictions, output_dict=True),
                    )

                    results_df = pd.concat(
                        [
                            results_df,
                            pd.DataFrame(
                                [
                                    {
                                        "task_id": task_id,
                                        "strategy": strategy,
                                        "accuracy": accuracy,
                                        "f1": f1,
                                        "dataset_id": task.dataset_id,
                                        "n_rows": n_rows_used,
                                    },
                                ],
                            ),
                        ],
                        ignore_index=True,
                    )

    results_df.to_excel("../benchmark_results/classification_results.xlsx", index=False)


if __name__ == "__main__":
    main()
