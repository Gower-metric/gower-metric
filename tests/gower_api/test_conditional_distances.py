import numpy as np
import pytest

from gower_metric import Gower


@pytest.mark.asyncio
async def test_conditional_distances() -> None:
    raw = np.array(
        [
            ["A", 0.0],
            ["A", 2.0],
            ["B", 5.0],
            ["B", 7.0],
        ],
        dtype=object,
    )
    f_types = {0: "categorical_nominal", 1: "numeric"}

    gower = Gower(
        feature_types=f_types,
        conditional_distances=True,
    ).fit(raw)

    # max - min
    r_max_min = 7 - 0

    # Manhattan distance, normalized by range
    def d(x, y):
        return abs(x - y) / r_max_min

    expected = np.array(
        [
            # 0, 1 , 2, 3
            [d(0, 0), d(0, 2), d(0, 5), d(0, 7)],  # 0
            [d(2, 0), d(2, 2), d(2, 5), d(2, 7)],  # 1
            [d(5, 0), d(5, 2), d(5, 5), d(5, 7)],  # 2
            [d(7, 0), d(7, 2), d(7, 5), d(7, 7)],  # 3
        ],
        dtype=float,
    )

    for i in range(raw.shape[0]):
        for j in range(raw.shape[0]):
            dist = gower(raw[i], raw[j])

            assert pytest.approx(dist, rel=1e-6) == expected[i, j]


@pytest.mark.asyncio
async def test_conditional_distances_clip() -> None:
    raw = np.array(
        [
            ["A", "X", 0.0],
            ["A", "Y", 2.0],
            ["B", "X", 5.0],
            ["B", "Y", 7.0],
        ],
        dtype=object,
    )
    f_types = {0: "categorical_nominal", 1: "categorical_nominal", 2: "numeric"}

    gower = Gower(
        feature_types=f_types,
        conditional_distances=True,
    ).fit(raw)

    assert gower(raw[0], raw[3]) == 1.0
    assert gower(raw[1], raw[2]) == 1.0

    # max - min for the numeric feature
    r = 7 - 0

    # Manhattan distance, normalized by range
    def d(x, y):
        return abs(x - y) / r

    expected = np.array(
        [
            # 0, 1, 2, 3
            [0.0, d(0, 2), d(0, 5), 1.0],  # 0
            [d(2, 0), 0.0, 1.0, d(2, 7)],  # 1
            [d(5, 0), 1.0, 0.0, d(5, 7)],  # 2
            [1.0, d(7, 2), d(7, 5), 0.0],  # 3
        ],
        dtype=float,
    )

    for i in range(raw.shape[0]):
        for j in range(raw.shape[0]):
            dist = gower(raw[i], raw[j])

            assert pytest.approx(dist, rel=1e-6) == expected[i, j]
