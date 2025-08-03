import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import pairwise_distances

from gower_metric import Gower


@pytest.mark.asyncio
async def test_scikit_learn_paiwise_distances() -> None:
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

    feature_types = {
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

    def _gower_distance(x, y):
        """
        Compute Gower distance between two vectors.
        """
        return gower(x, y)

    matrix_scikit = pairwise_distances(
        df, metric=_gower_distance, n_jobs=-1, ensure_all_finite=False
    )

    matrix_custom = np.zeros((n_rows, n_rows))
    for i in range(n_rows):
        for j in range(n_rows):
            matrix_custom[i, j] = gower(df[i], df[j])

    assert matrix_scikit.shape == (n_rows, n_rows), (
        "The shape of the pairwise distance matrix is incorrect."
    )
    assert matrix_custom.shape == (n_rows, n_rows), (
        "The shape of the custom pairwise distance matrix is incorrect."
    )
    assert np.allclose(matrix_scikit, matrix_custom, rtol=1e-5, atol=1e-8), (
        "The pairwise distance matrices do not match."
    )
