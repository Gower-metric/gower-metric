import warnings
from typing import cast

import numpy as np
import pytest
from rpy2 import rinterface, robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from gower_metric import Config, Gower

from .conftest import EDUCATION_LEVELS, generate_adult_like_df, generate_mixed_df

warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

if not rinterface.initr():
    rinterface.initr()


@pytest.mark.parametrize("n", [50, 100, 200, 500, 1000, 2000])
def test_r_daisy_no_weights(n: int, random_seed: int) -> None:
    rng = np.random.default_rng(random_seed)
    df = generate_adult_like_df(n, rng)

    converter = robjects.default_converter + pandas2ri.converter
    with converter.context():
        r_df = robjects.conversion.get_conversion().py2rpy(df)

    importr("cluster")
    importr("base")

    daisy = robjects.r["daisy"]
    dist_matrix = daisy(r_df, metric="gower")

    as_matrix = robjects.r["as.matrix"]
    r_matrix = as_matrix(dist_matrix)
    np_matrix = np.array(r_matrix)

    assert np_matrix.shape == (n, n)

    # gower section
    feature_types: dict[int | str, str] = {
        "age": "ratio_scale_interval",
        "education_num": "ratio_scale_interval",
        "race": "categorical_nominal",
        "sex": "categorical_nominal",
        "hours_per_week": "ratio_scale_interval",
    }

    cfg = Config(
        feature_types=feature_types,
    )
    gower = Gower(cfg).fit(df)

    X = df.to_numpy()
    gower_matrix = gower.matrix(X)

    assert np.allclose(
        cast("np.ndarray", np_matrix),
        cast("np.ndarray", gower_matrix),
        atol=1e-6,
    ), (
        f"Matrices differ (seed={random_seed}, n={n}), max diff={np.max(np.abs(gower_matrix - np_matrix))}"
    )


@pytest.mark.parametrize("n", [20, 50, 100, 500, 1000, 2000])
def test_r_daisy_weights(n: int, random_seed: int) -> None:
    rng = np.random.default_rng(random_seed)
    df = generate_mixed_df(n, rng)

    df = df.rename(
        columns={
            "Age": "age",
            "Salary": "income",
            "Have_children": "married",
            "Is_smoking": "infected",
            "Birth": "gender",
            "Education": "education",
        },
    )

    feature_types: dict[int | str, str] = {
        "age": "ratio_scale_interval",
        "gender": "categorical_nominal",
        "income": "numeric",
        "education": "categorical_ordinal",
        "married": "binary_symmetric",
        "infected": "binary_asymmetric",
    }

    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
        "education": EDUCATION_LEVELS,
    }

    feature_weights = {
        0: 1.0,
        1: 2.0,
        2: 3.0,
        3: 4.0,
        4: 5.0,
        5: 6.0,
    }

    cfg = Config(
        feature_types=feature_types,
        feature_weights=feature_weights,
        scale_method="range",
        categorical_ordinal_values_order=categorical_ordinal_values_order,
    )
    gower = Gower(cfg).fit(df)

    matrix = gower.matrix(df)

    assert matrix.shape == (n, n)

    converter = robjects.default_converter + pandas2ri.converter
    with converter.context():
        r_df = robjects.conversion.get_conversion().py2rpy(df)

    colnames = list(r_df.colnames)

    robjects.r(
        "ordered <- function(x, levels) { factor(x, levels=levels, ordered=TRUE) }",
    )

    r_df[colnames.index("education")] = robjects.r["ordered"](
        r_df[colnames.index("education")],
        robjects.StrVector(EDUCATION_LEVELS),
    )

    r_df[colnames.index("married")] = robjects.r["factor"](
        r_df[colnames.index("married")],
        levels=robjects.IntVector([0, 1]),
    )

    r_df[colnames.index("infected")] = robjects.r["as.logical"](
        r_df[colnames.index("infected")],
    )

    r_df[colnames.index("gender")] = robjects.r["factor"](
        r_df[colnames.index("gender")],
    )

    importr("cluster")
    daisy = robjects.r["daisy"]

    type_list = robjects.ListVector(
        {
            "asymm": robjects.StrVector(["infected"]),
            "symm": robjects.StrVector(["married"]),
        },
    )

    r_weights = robjects.FloatVector([1, 2, 3, 4, 5, 6])
    dist_matrix = daisy(r_df, metric="gower", type=type_list, weights=r_weights)

    as_matrix = robjects.r["as.matrix"]
    r_matrix = as_matrix(dist_matrix)
    np_matrix = np.array(r_matrix)

    assert np_matrix.shape == (n, n)
    assert np.allclose(
        cast("np.ndarray", np_matrix),
        cast("np.ndarray", matrix),
        atol=1e-6,
    ), (
        f"Matrices differ (seed={random_seed}, n={n}), max diff={np.max(np.abs(matrix - np_matrix))}"
    )
