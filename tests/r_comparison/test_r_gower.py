import warnings

import numpy as np
import pandas as pd
from rpy2 import rinterface, robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from gower_metric import Config, Gower

warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

if not rinterface.initr():
    rinterface.initr()


def test_r_gower() -> None:
    iris = pd.read_csv("data/files/iris.csv")

    dat1 = iris.iloc[0:10].reset_index(drop=True)
    dat2 = iris.iloc[5:15].reset_index(drop=True)

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

    dist_py = np.array([gower(dat1.iloc[i], dat2.iloc[i]) for i in range(10)])

    gower = importr("gower")

    converter = robjects.default_converter + pandas2ri.converter
    with converter.context():
        r_dat1 = robjects.conversion.get_conversion().py2rpy(dat1)
        r_dat2 = robjects.conversion.get_conversion().py2rpy(dat2)
        r_dist = gower.gower_dist(r_dat1, r_dat2)

    r_dist = np.array(r_dist)

    r_dist = gower.gower_dist(r_dat1, r_dat2)
    r_dist = np.array(r_dist)

    assert dist_py.shape == r_dist.shape, (
        "Shape mismatch between Python and R distances"
    )
    assert np.allclose(dist_py, r_dist, atol=1e-6), (
        "Distances do not match between Python and R implementations"
    )
