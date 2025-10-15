import numpy as np
import pandas as pd

from gower_metric import Gower


def test_gower_matrix_endpoint_with_custom_created_matrix() -> None:
    n_rows = 500
    df = pd.read_csv("./comparison/data/adult.csv").head(n_rows)

    df = df[
        [
            "age",
            "educational-num",
            "race",
            "gender",
            "hours-per-week",
            "relationship",
            "occupation",
            "education",
            "workclass",
        ]
    ]

    df = df.replace(to_replace="?", value=np.nan)

    feature_types: dict[int | str, str] = {
        "age": "ratio_scale_interval",
        "educational-num": "ratio_scale_interval",
        "race": "categorical_nominal",
        "gender": "categorical_nominal",
        "hours-per-week": "ratio_scale_interval",
        "relationship": "categorical_nominal",
        "occupation": "categorical_nominal",
        "education": "categorical_nominal",
        "workclass": "categorical_nominal",
    }

    gower = Gower(feature_types=feature_types).fit(df)
    df = df.to_numpy()

    dist_matrix: np.ndarray = gower.matrix(df)

    assert dist_matrix.shape == (n_rows, n_rows), (
        f"Unexpected shape: {dist_matrix.shape}"
    )

    matrix_custom: np.ndarray = np.zeros((n_rows, n_rows), dtype=np.float32)
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            matrix_custom[i, j] = gower(df[i], df[j])
            matrix_custom[j, i] = matrix_custom[i, j]

    assert np.allclose(dist_matrix, matrix_custom, rtol=1e-5, atol=1e-5), (
        "Matrices are not equal"
    )


def test_gower_matrix_endpoint_similarity() -> None:
    n_rows = 500
    df = pd.read_csv("./comparison/data/adult.csv").head(n_rows)

    df = df[
        [
            "age",
            "educational-num",
            "race",
            "gender",
            "hours-per-week",
            "relationship",
            "occupation",
            "education",
            "workclass",
        ]
    ]

    df = df.replace(to_replace="?", value=np.nan)

    feature_types: dict[int | str, str] = {
        "age": "ratio_scale_interval",
        "educational-num": "ratio_scale_interval",
        "race": "categorical_nominal",
        "gender": "categorical_nominal",
        "hours-per-week": "ratio_scale_interval",
        "relationship": "categorical_nominal",
        "occupation": "categorical_nominal",
        "education": "categorical_nominal",
        "workclass": "categorical_nominal",
    }

    gower = Gower(feature_types=feature_types).fit(df)
    df = df.to_numpy()

    similarity_matrix: np.ndarray = gower.matrix(df, matrix_type="similarity")

    matrix_custom: np.ndarray = np.zeros((n_rows, n_rows), dtype=np.float32)
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            matrix_custom[i, j] = gower.similarity(df[i], df[j])
            matrix_custom[j, i] = matrix_custom[i, j]

    assert np.allclose(similarity_matrix, matrix_custom, rtol=1e-5, atol=1e-5), (
        "Matrices are not equal"
    )

    assert np.allclose(similarity_matrix, similarity_matrix.T, rtol=1e-5, atol=1e-5), (
        "Matrix is symmetrical"
    )


def test_gower_matrix_endpoint_if_it_symmetrical() -> None:
    n_rows = 500
    df = pd.read_csv("./comparison/data/adult.csv").head(n_rows)

    df = df[
        [
            "age",
            "educational-num",
            "race",
            "gender",
            "hours-per-week",
            "relationship",
            "occupation",
            "education",
            "workclass",
        ]
    ]

    df = df.replace(to_replace="?", value=np.nan)

    feature_types: dict[int | str, str] = {
        "age": "ratio_scale_interval",
        "educational-num": "ratio_scale_interval",
        "race": "categorical_nominal",
        "gender": "categorical_nominal",
        "hours-per-week": "ratio_scale_interval",
        "relationship": "categorical_nominal",
        "occupation": "categorical_nominal",
        "education": "categorical_nominal",
        "workclass": "categorical_nominal",
    }

    feature_weights = {
        0: 1.0,
        1: 2.0,
        2: 3.0,
        3: 4.0,
        4: 5.0,
        5: 6.0,
        6: 7.0,
        7: 8.0,
        8: 9.0,
    }

    gower = Gower(
        feature_types=feature_types,
        feature_weights=feature_weights,
        scale="iqr",
        scale_window="kde",
    ).fit(df)
    df = df.to_numpy()

    dist_matrix: np.ndarray = gower.matrix(df)

    assert dist_matrix.shape == (n_rows, n_rows), (
        f"Unexpected shape: {dist_matrix.shape}"
    )

    custom_matrix = np.zeros((n_rows, n_rows), dtype=np.float32)
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            custom_matrix[i, j] = gower(df[i], df[j])
            custom_matrix[j, i] = custom_matrix[i, j]

    assert np.allclose(dist_matrix, custom_matrix, rtol=1e-5, atol=1e-5), (
        "Matrices are not equal"
    )

    assert np.allclose(dist_matrix, dist_matrix.T, rtol=1e-5, atol=1e-5), (
        "Matrix is not symmetrical"
    )


def test_matrix_endpoint_podani_if_symmetrical_distance() -> None:
    data = np.array(
        [["low", "high"], ["medium", "high"], ["high", "high"], ["low", "high"]],
        dtype=object,
    )
    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
        0: ["low", "medium", "high"],
    }

    gower = Gower(
        {0: "categorical_ordinal", 1: "categorical_nominal"},
        categorical_ordinal_values_order=categorical_ordinal_values_order,
        categorical_ordinal_calculation_type="podani",
    ).fit(data)

    dist_matrix: np.ndarray = gower.matrix(data)

    custom_matrix = np.zeros((data.shape[0], data.shape[0]), dtype=np.float32)

    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            custom_matrix[i, j] = gower(data[i], data[j])
            custom_matrix[j, i] = custom_matrix[i, j]

    assert np.allclose(dist_matrix, custom_matrix, rtol=1e-5, atol=1e-5), (
        "Matrices are not equal"
    )

    assert np.allclose(dist_matrix, dist_matrix.T, rtol=1e-5, atol=1e-5), (
        "Matrix is not symmetrical"
    )


def test_matrix_endpoint_podani_if_symmetrical_similarity() -> None:
    data = np.array(
        [["low", "high"], ["medium", "high"], ["high", "high"], ["low", "high"]],
        dtype=object,
    )
    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
        0: ["low", "medium", "high"],
    }

    gower = Gower(
        {0: "categorical_ordinal", 1: "categorical_nominal"},
        categorical_ordinal_values_order=categorical_ordinal_values_order,
        categorical_ordinal_calculation_type="podani",
    ).fit(data)

    dist_matrix: np.ndarray = gower.matrix(data, matrix_type="similarity")

    custom_matrix = np.zeros((data.shape[0], data.shape[0]), dtype=np.float32)

    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            custom_matrix[i, j] = gower.similarity(data[i], data[j])
            custom_matrix[j, i] = custom_matrix[i, j]

    assert np.allclose(dist_matrix, custom_matrix, rtol=1e-5, atol=1e-5), (
        "Matrices are not equal"
    )

    assert np.allclose(dist_matrix, dist_matrix.T, rtol=1e-5, atol=1e-5), (
        "Matrix is not symmetrical"
    )
