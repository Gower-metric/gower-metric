import numpy as np
import pytest

from gower_similarity.core.similarity import GowerSimilarity

@pytest.mark.asyncio
async def test_categorical_ordinal_kaufman_uniform_ndarray():
    data = np.array([['low'], ['medium'], ['high'], ['low']], dtype=object)
    gs = GowerSimilarity({0: 'categorical_ordinal'})
    gs.fit(data)

    expected = np.array([
        [0.0, 0.5, 1.0, 0.0],
        [0.5, 0.0, 0.5, 0.5],
        [1.0, 0.5, 0.0, 1.0],
        [0.0, 0.5, 1.0, 0.0],
    ], dtype=float)

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            xi = data[i]
            xj = data[j]
            dist = gs.distance(xi, xj)
            sim  = gs.similarity(xi, xj)
            assert pytest.approx(dist, rel=1e-6) == expected[i, j]
            assert pytest.approx(sim,  rel=1e-6) == 1.0 - expected[i, j]


@pytest.mark.asyncio
async def test_categorical_ordinal_podani_uniform_ndarray():
    data = np.array([['low'], ['medium'], ['high'], ['low']], dtype=object)
    gs = GowerSimilarity(
        {0: 'categorical_ordinal'},
        categorical_ordinal_calculation_type='podani'
    )
    gs.fit(data)

    expected = np.array([
        [0.0, 1/3, 1.0, 0.0],
        [1/3, 0.0, 2/3, 1/3],
        [1.0, 2/3, 0.0, 1.0],
        [0.0, 1/3, 1.0, 0.0],
    ], dtype=float)

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            xi = data[i]
            xj = data[j]
            dist = gs.distance(xi, xj)
            sim  = gs.similarity(xi, xj)
            assert pytest.approx(dist, rel=1e-6) == expected[i, j]
            assert pytest.approx(sim,  rel=1e-6) == 1.0 - expected[i, j]


@pytest.mark.asyncio
async def test_categorical_ordinal_not_valid_uniform_ndarray_():
    data = np.array([['low'], ['medium'], ['high'], ['low']], dtype=object)

    with pytest.raises(ValueError) as excinfo:
        gs = GowerSimilarity(
            {0: 'categorical_ordinal'},
            categorical_ordinal_calculation_type='not_valid'
        )
        gs.fit(data)

# TODO: add tests with weights