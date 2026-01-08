import numpy as np
import pandas as pd
import scipy.sparse as sp

from gower_metric import Gower
from gower_metric.core.config import Config


def test_gower_matrix_endpoint_with_custom_created_matrix() -> None:
    n_rows = 500
    df = pd.read_csv("data/files/adult.csv").head(n_rows)

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

    cfg = Config(
        feature_types=feature_types,
    )
    gower = Gower(cfg).fit(df)

    df = df.to_numpy()

    dist_matrix: np.ndarray = gower.matrix(df, backend="loky")

    assert dist_matrix.shape == (n_rows, n_rows), (
        f"Unexpected shape: {dist_matrix.shape}"
    )

    matrix_custom: np.ndarray = np.zeros((n_rows, n_rows), dtype=np.float32)
    for i in range(n_rows):
        for j in range(n_rows):
            matrix_custom[i, j] = gower(df[i], df[j])

    assert np.allclose(dist_matrix, matrix_custom, rtol=1e-5, atol=1e-5), (
        "Matrices are not equal"
    )


def test_gower_matrix_endpoint_similarity() -> None:
    n_rows = 500
    df = pd.read_csv("data/files/adult.csv").head(n_rows)

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
    cfg = Config(
        feature_types=feature_types,
    )
    gower = Gower(cfg).fit(df)

    df = df.to_numpy()

    similarity_matrix: np.ndarray = gower.matrix(
        df,
        matrix_type="similarity",
        backend="loky",
    )

    matrix_custom: np.ndarray = np.zeros((n_rows, n_rows), dtype=np.float32)
    for i in range(n_rows):
        for j in range(n_rows):
            matrix_custom[i, j] = gower.similarity(df[i], df[j])

    assert np.allclose(similarity_matrix, matrix_custom, rtol=1e-5, atol=1e-5), (
        "Matrices are not equal"
    )

    assert np.allclose(similarity_matrix, similarity_matrix.T, rtol=1e-5, atol=1e-5), (
        "Matrix is symmetrical"
    )


def test_gower_matrix_endpoint_if_it_symmetrical() -> None:
    n_rows = 500
    df = pd.read_csv("data/files/adult.csv").head(n_rows)

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

    cfg = Config(
        feature_types=feature_types,
        feature_weights=feature_weights,
        scale_method="iqr",
        scale_window="kde",
    )
    gower = Gower(cfg).fit(df)

    df = df.to_numpy()

    dist_matrix: np.ndarray = gower.matrix(df, backend="loky")

    assert dist_matrix.shape == (n_rows, n_rows), (
        f"Unexpected shape: {dist_matrix.shape}"
    )

    custom_matrix = np.zeros((n_rows, n_rows), dtype=np.float32)
    for i in range(n_rows):
        for j in range(n_rows):
            custom_matrix[i, j] = gower(df[i], df[j])

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

    cfg = Config(
        feature_types={0: "categorical_ordinal", 1: "categorical_nominal"},
        categorical_ordinal_values_order=categorical_ordinal_values_order,
        categorical_ordinal_calculation_type="podani",
    )
    gower = Gower(cfg).fit(data)

    dist_matrix: np.ndarray = gower.matrix(data, backend="loky")

    n = data.shape[0]
    custom_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            custom_matrix[i, j] = gower(data[i], data[j])

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

    cfg = Config(
        feature_types={0: "categorical_ordinal", 1: "categorical_nominal"},
        categorical_ordinal_values_order=categorical_ordinal_values_order,
        categorical_ordinal_calculation_type="podani",
    )
    gower = Gower(cfg).fit(data)

    dist_matrix: np.ndarray = gower.matrix(
        data,
        matrix_type="similarity",
        backend="loky",
    )

    n = data.shape[0]
    custom_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            custom_matrix[i, j] = gower.similarity(data[i], data[j])

    assert np.allclose(dist_matrix, custom_matrix, rtol=1e-5, atol=1e-5), (
        "Matrices are not equal"
    )

    assert np.allclose(dist_matrix, dist_matrix.T, rtol=1e-5, atol=1e-5), (
        "Matrix is not symmetrical"
    )


def test_sparse_matrix_convertion_csr() -> None:
    data = np.array(
        [["low", "high"], ["medium", "high"], ["high", "high"], ["low", "high"]],
        dtype=object,
    )
    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
        0: ["low", "medium", "high"],
    }

    cfg = Config(
        feature_types={0: "categorical_ordinal", 1: "categorical_nominal"},
        categorical_ordinal_values_order=categorical_ordinal_values_order,
        categorical_ordinal_calculation_type="podani",
    )
    gower = Gower(cfg).fit(data)

    dist_matrix: np.ndarray = gower.matrix(
        data,
        convert_to_sparse=True,
        sparse_type="csr",
        backend="loky",
    )

    assert sp.issparse(dist_matrix), "Matrix is not sparse"
    assert sp.isspmatrix_csr(dist_matrix), "Matrix is not csr format"


def test_sparse_matrix_convertion_csc() -> None:
    data = np.array(
        [["low", "high"], ["medium", "high"], ["high", "high"], ["low", "high"]],
        dtype=object,
    )
    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
        0: ["low", "medium", "high"],
    }

    cfg = Config(
        feature_types={0: "categorical_ordinal", 1: "categorical_nominal"},
        categorical_ordinal_values_order=categorical_ordinal_values_order,
        categorical_ordinal_calculation_type="podani",
    )
    gower = Gower(cfg).fit(data)

    dist_matrix: np.ndarray = gower.matrix(
        data,
        convert_to_sparse=True,
        sparse_type="csc",
        backend="loky",
    )

    assert sp.issparse(dist_matrix), "Matrix is not sparse"
    assert sp.isspmatrix_csc(dist_matrix), "Matrix is not csc format"


def test_sparse_matrix_convertion_c00() -> None:
    data = np.array(
        [["low", "high"], ["medium", "high"], ["high", "high"], ["low", "high"]],
        dtype=object,
    )
    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
        0: ["low", "medium", "high"],
    }

    cfg = Config(
        feature_types={0: "categorical_ordinal", 1: "categorical_nominal"},
        categorical_ordinal_values_order=categorical_ordinal_values_order,
        categorical_ordinal_calculation_type="podani",
    )
    gower = Gower(cfg).fit(data)

    dist_matrix: np.ndarray = gower.matrix(
        data,
        convert_to_sparse=True,
        sparse_type="coo",
        backend="loky",
    )

    assert sp.issparse(dist_matrix), "Matrix is not sparse"
    assert sp.isspmatrix_coo(dist_matrix), "Matrix is not coo format"
