import numpy as np
import pandas as pd
import pytest
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from gower_similarity.core.similarity import GowerSimilarity


@pytest.mark.asyncio
async def test_r_gower() -> None:
    iris = pd.read_csv("./comparison/data/iris.csv")

    dat1 = iris.iloc[0:10].reset_index(drop=True)
    dat2 = iris.iloc[5:15].reset_index(drop=True)

    f_types = {
        "sepal_length": "numeric",
        "sepal_width": "numeric",
        "petal_length": "numeric",
        "petal_width": "numeric",
        "variety": "categorical_nominal",
    }

    union = pd.concat([dat1, dat2], axis=0, ignore_index=True)
    gs = GowerSimilarity(feature_types=f_types, scale="range").fit(union)

    dist_py = np.array([gs.distance(dat1.iloc[i], dat2.iloc[i]) for i in range(10)])

    gower = importr("gower")
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_dat1 = robjects.conversion.py2rpy(dat1)
        r_dat2 = robjects.conversion.py2rpy(dat2)

    r_dist = gower.gower_dist(r_dat1, r_dat2)
    r_dist = np.array(r_dist)

    assert dist_py.shape == r_dist.shape, (
        "Shape mismatch between Python and R distances"
    )
    assert np.allclose(dist_py, r_dist, atol=1e-6), (
        "Distances do not match between Python and R implementations"
    )
