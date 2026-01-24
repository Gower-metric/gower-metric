from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from gower_metric import Config, Gower

DTYPE = np.float64


def test_scikit_learn_paiwise_distances() -> None:
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
        data_type=DTYPE,
    )
    gower = Gower(cfg).fit(df)
    transformed_df = gower.transform(df)

    matrix_scikit = pairwise_distances(
        transformed_df,
        metric=gower,
        n_jobs=-1,
        ensure_all_finite=False,
    )

    matrix_gower = gower.matrix(transformed_df, backend="loky")

    assert matrix_scikit.shape == (n_rows, n_rows), (
        "The shape of the pairwise distance matrix is incorrect."
    )
    assert matrix_gower.shape == (n_rows, n_rows), (
        "The shape of the custom pairwise distance matrix is incorrect."
    )
    assert np.allclose(
        cast("np.ndarray", matrix_scikit),
        cast("np.ndarray", matrix_gower),
        rtol=1e-5,
        atol=1e-8,
    ), "The pairwise distance matrices do not match."
