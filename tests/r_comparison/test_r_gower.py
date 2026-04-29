import warnings

import numpy as np
import pandas as pd
import pytest
from rpy2 import rinterface, robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from gower_metric import Config, Gower

from .conftest import generate_numeric_with_categorical_df

warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

if not rinterface.initr():
    rinterface.initr()


@pytest.mark.parametrize("n", [20, 50, 100, 500, 1000, 2000])
def test_r_gower(n: int, random_seed: int) -> None:
    rng = np.random.default_rng(random_seed)
    full_df = generate_numeric_with_categorical_df(n * 2, rng)

    dat1 = full_df.iloc[:n].reset_index(drop=True)
    dat2 = full_df.iloc[n : n * 2].reset_index(drop=True)

    f_types: dict[int | str, str] = {
        "sepal_length": "numeric",
        "sepal_width": "numeric",
        "petal_length": "numeric",
        "petal_width": "numeric",
        "variety": "categorical_nominal",
    }

    union = pd.concat([dat1, dat2], axis=0, ignore_index=True)

    cfg = Config(
        feature_types=f_types,
        scale_method="range",
    )
    gower = Gower(cfg).fit(union)

    dist_py = np.array([gower(dat1.iloc[i], dat2.iloc[i]) for i in range(n)])

    gower_r = importr("gower")

    converter = robjects.default_converter + pandas2ri.converter
    with converter.context():
        r_dat1 = robjects.conversion.get_conversion().py2rpy(dat1)
        r_dat2 = robjects.conversion.get_conversion().py2rpy(dat2)
        r_dist = gower_r.gower_dist(r_dat1, r_dat2)

    r_dist = np.array(r_dist)

    assert dist_py.shape == r_dist.shape, (
        f"Shape mismatch (seed={random_seed}, n={n}): Python {dist_py.shape} vs R {r_dist.shape}"
    )
    assert np.allclose(dist_py, r_dist, atol=1e-6), (
        f"Distances differ (seed={random_seed}, n={n}), max diff={np.max(np.abs(dist_py - r_dist))}"
    )
