import warnings

import numpy as np
import pandas as pd
import pytest
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from gower_metric import Gower

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.mark.asyncio
async def test_r_daisy_no_weights() -> None:
    n_rows = 100
    df = pd.read_csv("./comparison/data/adult_reduced.csv").head(n_rows)
    df["race"] = df["race"].astype("category")
    df["sex"] = df["sex"].astype("category")

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = robjects.conversion.py2rpy(df)

    importr("cluster")
    importr("base")

    daisy = robjects.r["daisy"]
    dist_matrix = daisy(r_df, metric="gower")

    as_matrix = robjects.r["as.matrix"]
    r_matrix = as_matrix(dist_matrix)
    np_matrix = np.array(r_matrix)

    assert np_matrix.shape == (n_rows, n_rows)

    # gower section
    feature_types: dict[int | str, str] = {
        "age": "ratio_scale_interval",
        "education_num": "ratio_scale_interval",
        "race": "categorical_nominal",
        "sex": "categorical_nominal",
        "hours_per_week": "ratio_scale_interval",
    }

    gower = Gower(feature_types=feature_types).fit(df)

    gower_matrix = np.zeros((n_rows, n_rows), dtype=np.float32)
    df = df.to_numpy()

    for i in range(n_rows):
        for j in range(n_rows):
            gower_matrix[i, j] = gower(df[i], df[j])

    assert np.allclose(np_matrix, gower_matrix, atol=1e-6)


@pytest.mark.asyncio
async def test_r_daisy_weights() -> None:
    df = pd.DataFrame(
        {
            "age": [23, 45, 23, 31],
            "gender": ["Female", "Male", "Female", "Male"],
            "income": [35000, 81000, 40000, 30000],
            "education": ["low", "medium", "high", "low"],
            "married": [0, 1, 1, 0],
            "infected": [1, 1, 0, 0],
        }
    )

    feature_types: dict[int | str, str] = {
        "age": "ratio_scale_interval",
        "gender": "categorical_nominal",
        "income": "numeric",
        "education": "categorical_ordinal",
        "married": "binary_symmetric",
        "infected": "binary_asymmetric",
    }

    feature_weights: dict[int, float] | str | None = {
        0: 1.0,
        1: 2.0,
        2: 3.0,
        3: 4.0,
        4: 5.0,
        5: 6.0,
    }

    gower = Gower(
        feature_types=feature_types, feature_weights=feature_weights, scale="range"
    ).fit(df)

    n = len(df)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = gower(df.iloc[i], df.iloc[j])

    assert matrix.shape == (n, n)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = robjects.conversion.py2rpy(df)

    colnames = list(r_df.colnames)

    robjects.r(
        "ordered <- function(x, levels) { factor(x, levels=levels, ordered=TRUE) }"
    )

    r_df[colnames.index("education")] = robjects.r["ordered"](
        r_df[colnames.index("education")], robjects.StrVector(["low", "medium", "high"])
    )

    r_df[colnames.index("married")] = robjects.r["factor"](
        r_df[colnames.index("married")], levels=robjects.IntVector([0, 1])
    )

    r_df[colnames.index("infected")] = robjects.r["as.logical"](
        r_df[colnames.index("infected")]
    )

    r_df[colnames.index("gender")] = robjects.r["factor"](
        r_df[colnames.index("gender")]
    )

    importr("cluster")
    daisy = robjects.r["daisy"]

    weights = robjects.FloatVector([1, 2, 3, 4, 5, 6])
    dist_matrix = daisy(r_df, metric="gower", weights=weights)

    as_matrix = robjects.r["as.matrix"]
    r_matrix = as_matrix(dist_matrix)
    np_matrix = np.array(r_matrix)

    assert np_matrix.shape == (n, n)
    assert np.allclose(np_matrix, matrix, atol=1e-6)
