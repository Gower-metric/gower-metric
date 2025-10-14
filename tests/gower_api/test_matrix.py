import numpy as np
import pandas as pd
import pytest

from gower_metric import Gower


@pytest.mark.asyncio
async def test_scikit_learn_paiwise_distances() -> None:
    n_rows = 1000
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

    dist_matrix: np.ndarray = gower.matrix(df, verbose=1)

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
