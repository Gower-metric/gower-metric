import numpy as np
import pandas as pd
import pytest

from gower_metric import Gower


@pytest.mark.asyncio
async def test_binary_symmetric_only() -> None:
    data = np.array([[0], [1], [0]], dtype=object)
    gower = Gower({0: "binary_symmetric"})
    gower.fit(data)

    # (0,0) -> distance 0, similarity 1
    assert pytest.approx(gower(data[0], data[2]), rel=1e-6) == 0.0

    # (1,1) -> distance 0, similarity 1
    assert pytest.approx(gower(data[1], data[1]), rel=1e-6) == 0.0

    # (0,1) and (1,0) -> distance 1, similarity 0
    assert pytest.approx(gower(data[0], data[1]), rel=1e-6) == 1.0
    assert pytest.approx(gower(data[1], data[0]), rel=1e-6) == 1.0


@pytest.mark.asyncio
async def test_binary_symmetric_only_pandas() -> None:
    data = pd.DataFrame({"column": [0, 1, 0]})
    gower = Gower({"column": "binary_symmetric"})
    gower.fit(data)

    # (0,0) -> distance 0, similarity 1
    assert pytest.approx(gower(data.iloc[0], data.iloc[2]), rel=1e-6) == 0.0

    # (1,1) -> distance 0, similarity 1
    assert pytest.approx(gower(data.iloc[1], data.iloc[1]), rel=1e-6) == 0.0

    # (0,1) and (1,0) -> distance 1, similarity 0
    assert pytest.approx(gower(data.iloc[0], data.iloc[1]), rel=1e-6) == 1.0
    assert pytest.approx(gower(data.iloc[1], data.iloc[0]), rel=1e-6) == 1.0
