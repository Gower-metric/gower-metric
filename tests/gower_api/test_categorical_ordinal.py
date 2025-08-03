import numpy as np
import pytest

from gower_metric import Gower


@pytest.mark.asyncio
async def test_categorical_ordinal_kaufman_uniform_ndarray() -> None:
    data = np.array([["low"], ["medium"], ["high"], ["low"]], dtype=object)
    gower = Gower({0: "categorical_ordinal"})
    gower.fit(data)

    expected = np.array(
        [
            [0.0, 0.5, 1.0, 0.0],
            [0.5, 0.0, 0.5, 0.5],
            [1.0, 0.5, 0.0, 1.0],
            [0.0, 0.5, 1.0, 0.0],
        ],
        dtype=float,
    )

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            xi = data[i]
            xj = data[j]
            dist = gower(xi, xj)
            sim = gower.similarity(xi, xj)
            assert pytest.approx(dist, rel=1e-6) == expected[i, j]
            assert pytest.approx(sim, rel=1e-6) == 1.0 - expected[i, j]


@pytest.mark.asyncio
async def test_categorical_ordinal_podani_uniform_ndarray() -> None:
    data = np.array([["low"], ["medium"], ["high"], ["low"]], dtype=object)
    gower = Gower(
        {0: "categorical_ordinal"}, categorical_ordinal_calculation_type="podani"
    )
    gower.fit(data)

    expected = np.array(
        [
            [0.0, 1 / 3, 1.0, 0.0],
            [1 / 3, 0.0, 2 / 3, 1 / 3],
            [1.0, 2 / 3, 0.0, 1.0],
            [0.0, 1 / 3, 1.0, 0.0],
        ],
        dtype=float,
    )

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            xi = data[i]
            xj = data[j]
            dist = gower(xi, xj)
            sim = gower.similarity(xi, xj)
            assert pytest.approx(dist, rel=1e-6) == expected[i, j]
            assert pytest.approx(sim, rel=1e-6) == 1.0 - expected[i, j]


@pytest.mark.asyncio
async def test_categorical_ordinal_not_valid_uniform_ndarray_() -> None:
    data = np.array([["low"], ["medium"], ["high"], ["low"]], dtype=object)

    with pytest.raises(ValueError):
        gower = Gower(
            {0: "categorical_ordinal"}, categorical_ordinal_calculation_type="not_valid"
        )
        gower.fit(data)
