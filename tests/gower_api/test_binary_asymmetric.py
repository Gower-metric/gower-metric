import numpy as np
import pytest

from gower_similarity.core.similarity import GowerSimilarity

@pytest.mark.asyncio
async def test_binary_asymmetric_only():
    data = np.array([[0], [1], [0]], dtype=object)
    gs = GowerSimilarity({0: 'binary_asymmetric'})
    gs.fit(data)

    # 0/0: NaN
    assert np.isnan(gs.distance(data[0], data[2]))

    # 1/1: distance 0
    assert pytest.approx(gs.distance(data[1], data[1]), rel=1e-6) == 0.0

    # 1/0: distance 1
    assert pytest.approx(gs.distance(data[1], data[0]), rel=1e-6) == 1.0
    assert pytest.approx(gs.similarity(data[1], data[0]), rel=1e-6) == 0.0

# TODO: add more complex tests