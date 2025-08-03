import numpy as np
import pandas as pd
import pytest

from gower_metric import Gower


@pytest.mark.asyncio
async def test_categorical_nominal_ndarray() -> None:
    data = np.array([["A"], ["B"], ["C"], ["A"]], dtype=object)
    gower = Gower({0: "categorical_nominal"})
    gower.fit(data)

    # A, B, C, A
    expected = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],  # A
            [1.0, 0.0, 1.0, 1.0],  # B
            [1.0, 1.0, 0.0, 1.0],  # C
            [0.0, 1.0, 1.0, 0.0],  # A
        ],
        dtype=float,
    )

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            xi = data[i]
            xj = data[j]
            dist = gower(xi, xj)
            assert pytest.approx(dist, rel=1e-6) == expected[i, j]


@pytest.mark.asyncio
async def test_categorical_nominal_pandas() -> None:
    df = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
    gower = Gower({"color": "categorical_nominal"})
    gower.fit(df)

    # red, blue, green, red
    expected = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],  # red
            [1.0, 0.0, 1.0, 1.0],  # blue
            [1.0, 1.0, 0.0, 1.0],  # green
            [0.0, 1.0, 1.0, 0.0],  # red
        ],
        dtype=float,
    )

    for i in df.index:
        for j in df.index:
            dist = gower(df.loc[i], df.loc[j])
            assert pytest.approx(dist, rel=1e-6) == expected[i, j]
