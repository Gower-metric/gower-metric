import numpy as np
import pandas as pd
import pytest

from gower_similarity.core.similarity import GowerSimilarity

@pytest.mark.asyncio
async def test_ratio_scale_ndarray():
    data = np.array([[1.0], [2.0], [3.0], [1.0]], dtype=float)
    gs = GowerSimilarity({0: 'ratio_scale_interval'})
    gs.fit(data)

    # Range: max - min = 3 - 1 = 2 -> so matematically it will be |x - y| / 2

    # 1, 2, 3, 1
    expected = np.array(
        [
            [0.0, 0.5, 1.0, 0.0],  # 1
            [0.5, 0.0, 0.5, 0.5],  # 2
            [1.0, 0.5, 0.0, 1.0],  # 3
            [0.0, 0.5, 1.0, 0.0],  # 1
        ],
        dtype=float)

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            dist = gs.distance(data[i], data[j])
            sim = gs.similarity(data[i], data[j])
            assert pytest.approx(dist, rel=1e-6) == expected[i, j]
            assert pytest.approx(sim, rel=1e-6) == 1.0 - expected[i, j]
