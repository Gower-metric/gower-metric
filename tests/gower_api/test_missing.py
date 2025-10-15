import numpy as np
import pandas as pd
import pytest

from gower_metric import Gower


def test_missing_values_skip_nan() -> None:
    """
    Test that missing values are handled correctly (ignored) in Gower similarity.
    """
    data = [
        [np.nan, "A"],
        [10.0, "A"],
        [np.nan, None],
    ]
    df = pd.DataFrame(data, columns=["num", "cat"])
    feature_types: dict[int | str, str] = {
        "num": "numeric",
        "cat": "categorical_nominal",
    }

    gower = Gower(feature_types).fit(df)

    assert pytest.approx(gower(df.iloc[0], df.iloc[1]), rel=1e-6) == 0.0
    assert np.isnan(gower(df.iloc[0], df.iloc[2]))
