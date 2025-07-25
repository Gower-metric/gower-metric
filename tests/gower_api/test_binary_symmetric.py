import numpy as np
import pandas as pd
import pytest

from gower_similarity.core.similarity import GowerSimilarity


@pytest.mark.asyncio
async def test_binary_symmetric_only() -> None:
    data = np.array([[0], [1], [0]], dtype=object)
    gs = GowerSimilarity({0: "binary_symmetric"})
    gs.fit(data)

    # (0,0) -> distance 0, similarity 1
    assert pytest.approx(gs.distance(data[0], data[2]), rel=1e-6) == 0.0
    assert pytest.approx(gs.similarity(data[0], data[2]), rel=1e-6) == 1.0

    # (1,1) -> distance 0, similarity 1
    assert pytest.approx(gs.distance(data[1], data[1]), rel=1e-6) == 0.0
    assert pytest.approx(gs.similarity(data[1], data[1]), rel=1e-6) == 1.0

    # (0,1) and (1,0) -> distance 1, similarity 0
    assert pytest.approx(gs.distance(data[0], data[1]), rel=1e-6) == 1.0
    assert pytest.approx(gs.similarity(data[0], data[1]), rel=1e-6) == 0.0
    assert pytest.approx(gs.distance(data[1], data[0]), rel=1e-6) == 1.0
    assert pytest.approx(gs.similarity(data[1], data[0]), rel=1e-6) == 0.0


@pytest.mark.asyncio
async def test_binary_symmetric_only_pandas() -> None:
    data = pd.DataFrame({"column": [0, 1, 0]})
    gs = GowerSimilarity({"column": "binary_symmetric"})
    gs.fit(data)

    # (0,0) -> distance 0, similarity 1
    assert pytest.approx(gs.distance(data.iloc[0], data.iloc[2]), rel=1e-6) == 0.0
    assert pytest.approx(gs.similarity(data.iloc[0], data.iloc[2]), rel=1e-6) == 1.0

    # (1,1) -> distance 0, similarity 1
    assert pytest.approx(gs.distance(data.iloc[1], data.iloc[1]), rel=1e-6) == 0.0
    assert pytest.approx(gs.similarity(data.iloc[1], data.iloc[1]), rel=1e-6) == 1.0

    # (0,1) and (1,0) -> distance 1, similarity 0
    assert pytest.approx(gs.distance(data.iloc[0], data.iloc[1]), rel=1e-6) == 1.0
    assert pytest.approx(gs.similarity(data.iloc[0], data.iloc[1]), rel=1e-6) == 0.0
    assert pytest.approx(gs.distance(data.iloc[1], data.iloc[0]), rel=1e-6) == 1.0
    assert pytest.approx(gs.similarity(data.iloc[1], data.iloc[0]), rel=1e-6) == 0.0
