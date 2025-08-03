import itertools

import numpy as np
import pytest

from gower_metric import Gower
from gower_metric.utils.kde_types.silverman import silverman_bandwidth


@pytest.mark.asyncio
async def test_ratio_scale_range_ndarray() -> None:
    data = np.array([[1.0], [2.0], [3.0], [1.0]], dtype=float)
    gower = Gower({0: "ratio_scale_interval"}, scale="range")
    gower.fit(data)

    # Range: max - min = 3 - 1 = 2 -> so matematically it will be |x - y| / 2

    # 1, 2, 3, 1
    expected = np.array(
        [
            [0.0, 0.5, 1.0, 0.0],  # 1
            [0.5, 0.0, 0.5, 0.5],  # 2
            [1.0, 0.5, 0.0, 1.0],  # 3
            [0.0, 0.5, 1.0, 0.0],  # 1
        ],
        dtype=float,
    )

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            dist = gower(data[i], data[j])
            sim = gower.similarity(data[i], data[j])
            assert pytest.approx(dist, rel=1e-6) == expected[i, j]
            assert pytest.approx(sim, rel=1e-6) == 1.0 - expected[i, j]


@pytest.mark.asyncio
async def test_ratio_scale_kde_window_h() -> None:
    # result >> h
    data = np.array([[0.0], [100.0], [200.0]], dtype=float)

    col = data[:, 0]
    manual_h = silverman_bandwidth(col)

    gs_kde = Gower(
        {0: "ratio_scale_interval"},
        scale="range",
        scale_window="kde",
        scale_window_type="silverman",
    )
    gs_kde.fit(data)

    assert isinstance(gs_kde._h_ratio, np.ndarray)
    assert gs_kde._h_ratio.shape == (1,)
    assert pytest.approx(gs_kde._h_ratio[0], rel=1e-6) == manual_h

    gs_plain = Gower({0: "ratio_scale_interval"}, scale="range")
    gs_plain.fit(data)

    for xi, xj in itertools.product(data, data):
        d1 = gs_plain(xi, xj)
        d2 = gs_kde(xi, xj)
        assert pytest.approx(d2, rel=1e-6) == d1
