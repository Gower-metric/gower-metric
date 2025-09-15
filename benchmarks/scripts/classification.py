import warnings

import openml
import pandas as pd
from joblib import parallel_backend
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)


def _load_data(dataset_id: int) -> tuple[pd.DataFrame, pd.Series]:
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    return X, y


def _split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _impute_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame
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
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])

        X_train = X_train.drop(columns=categorical_cols)
        X_test = X_test.drop(columns=categorical_cols)

        X_train = pd.concat(
            [X_train.reset_index(drop=True), pd.DataFrame(X_train_encoded)], axis=1
        )
        X_test = pd.concat(
            [X_test.reset_index(drop=True), pd.DataFrame(X_test_encoded)], axis=1
        )
    else:
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

    return X_train, X_test


def _grid_search_knn(X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:
    param_grid = {"n_neighbors": [3, 5, 7, 9, 11]}
    knn = KNeighborsClassifier(n_jobs=-1)
    grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, scoring="accuracy")

    with parallel_backend("multiprocessing"):
        grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


def main() -> None:
    suite = openml.study.get_suite(99)
    tasks = suite.tasks

    results = []
    results_f1 = []

    for task_id in tqdm(tasks, desc="Processing tasks", unit="task"):
        task = openml.tasks.get_task(task_id)

        X, y = _load_data(task.dataset_id)

        X_train, X_test, y_train, y_test = _split_data(X, y)
        X_train, X_test = _impute_data(X_train, X_test)
        X_train, X_test = _encode_data(X_train, X_test)

        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        knn = _grid_search_knn(X_train, y_train)
        knn.fit(X_train, y_train)

        prefictions = knn.predict(X_test)
        accuracy = accuracy_score(y_test, prefictions)
        results.append(accuracy)

        f1 = f1_score(y_test, prefictions, average="weighted")
        results_f1.append(f1)

    print(
        f"Average accuracy over {len(results)} tasks: {sum(results) / len(results):.4f}"
    )
    print(f"Standard deviation of accuracy: {pd.Series(results).std():.4f}")

    print(f"Average F1 score over tasks: {sum(results_f1) / len(results_f1):.4f}")
    print(f"Standard deviation of F1 score: {pd.Series(results_f1).std():.4f}")


if __name__ == "__main__":
    main()
