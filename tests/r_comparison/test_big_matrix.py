import warnings
from typing import cast

import numpy as np
import pytest
from rpy2 import rinterface, robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from gower_metric import Config, Gower

from .conftest import generate_mixed_df

warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

if not rinterface.initr():
    rinterface.initr()


@pytest.mark.parametrize("n", [50, 200, 500, 1000, 2000])
def test_big_matrix(n: int, random_seed: int) -> None:
    rng = np.random.default_rng(random_seed)
    df = generate_mixed_df(n, rng)

    features: dict[int | str, str] = {
        "Age": "numeric",
        "Salary": "ratio_scale_interval",
        "Have_children": "binary_symmetric",
        "Is_smoking": "binary_asymmetric",
        "Birth": "categorical_nominal",
        "Education": "categorical_ordinal",
    }
    ord_order: dict[int | str, list[str]] = {
        "Education": ["Low", "Medium", "High", "PhD", "Prof"],
    }

    weights = {0: 1.0, 1: 2.0, 2: 1.5, 3: 1.2, 4: 3.75, 5: 2.72}

    cfg = Config(
        feature_types=features,
        feature_weights=weights,
        categorical_ordinal_values_order=ord_order,
    )
    gower = Gower(cfg).fit(df)

    gower_matrix = gower.matrix(df)

    # R section
    converter = robjects.default_converter + pandas2ri.converter
    with converter.context():
        r_df = robjects.conversion.get_conversion().py2rpy(df)

    colnames = list(r_df.colnames)

    robjects.r(
        "ordered_f <- function(x, levels) { factor(x, levels=levels, ordered=TRUE) }",
    )
    robjects.r("factor_f <- function(x) { factor(x) }")
    robjects.r("logical_f <- function(x) { as.logical(x) }")

    # Education -> Ordered Factor
    r_df[colnames.index("Education")] = robjects.r["ordered_f"](
        r_df[colnames.index("Education")],
        robjects.StrVector(["Low", "Medium", "High", "PhD", "Prof"]),
    )

    # Have_children -> Factor (Binary Symmetric)
    r_df[colnames.index("Have_children")] = robjects.r["factor_f"](
        r_df[colnames.index("Have_children")],
    )

    # Is_smoking -> Logical (Binary Asymmetric)
    r_df[colnames.index("Is_smoking")] = robjects.r["logical_f"](
        r_df[colnames.index("Is_smoking")],
    )

    # Birth -> Factor (Nominal)
    r_df[colnames.index("Birth")] = robjects.r["factor_f"](
        r_df[colnames.index("Birth")],
    )

    importr("cluster")
    daisy = robjects.r["daisy"]

    type_list = robjects.ListVector(
        {
            "asymm": robjects.StrVector(["Is_smoking"]),
            "symm": robjects.StrVector(["Have_children"]),
        },
    )

    r_weights = robjects.FloatVector([1.0, 2.0, 1.5, 1.2, 3.75, 2.72])
    dist_matrix = daisy(r_df, metric="gower", type=type_list, weights=r_weights)

    as_matrix = robjects.r["as.matrix"]
    r_matrix = as_matrix(dist_matrix)
    np_matrix = np.array(r_matrix)

    assert np_matrix.shape == (n, n)
    assert np.allclose(
        cast("np.ndarray", np_matrix),
        cast("np.ndarray", gower_matrix),
        atol=1e-6,
    ), (
        f"Matrices differ (seed={random_seed}, n={n}), max diff={np.max(np.abs(gower_matrix - np_matrix))}"
    )
