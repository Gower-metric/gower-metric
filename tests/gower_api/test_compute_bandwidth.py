import numpy as np
import pytest

from gower_similarity.core.similarity import GowerSimilarity
from gower_similarity.utils.kde_types.silverman import silverman_bandwidth
from gower_similarity.utils.knn_bandwidth import knn_bandwidth


@pytest.mark.asyncio
async def test_ratio_scale_knn_window_no_error() -> None:
    rng = np.random.default_rng(seed=42)
    data = rng.normal(size=(60, 2))

    gs_knn = GowerSimilarity(
        {0: "ratio_scale_interval", 1: "numeric"},
        scale_window="kNN",
        scale="range",
    ).fit(data)

    assert isinstance(gs_knn._h_ratio, np.ndarray)
    assert (gs_knn._h_ratio > 0).all()
    assert (gs_knn._h_numeric > 0).all()


@pytest.mark.asyncio
async def test_ratio_scale_kde_window_h_multi() -> None:
    rng = np.random.default_rng(seed=123)
    data = rng.normal(size=(80, 2))

    gs_kde = GowerSimilarity(
        {0: "ratio_scale_interval", 1: "numeric"},
        scale_window="kde",
        scale_window_type="silverman",
        scale="range",
    ).fit(data)

    assert isinstance(gs_kde._h_ratio, np.ndarray)
    assert gs_kde._h_ratio.shape == (1,)
    assert isinstance(gs_kde._h_numeric, np.ndarray)
    assert gs_kde._h_numeric.shape == (1,)

    manual_h_ratio = silverman_bandwidth(data[:, 0])
    manual_h_numeric = silverman_bandwidth(data[:, 1])

    assert pytest.approx(gs_kde._h_ratio[0], rel=1e-12) == manual_h_ratio
    assert pytest.approx(gs_kde._h_numeric[0], rel=1e-12) == manual_h_numeric


@pytest.mark.asyncio
async def test_knn_bandwidth_values_and_effect() -> None:
    data = np.array(
        [
            [0.00, 0.00],
            [0.10, 0.05],
            [2.00, 1.00],
        ],
        dtype=object,
    )

    k = 1
    gs = GowerSimilarity(
        {0: "ratio_scale_interval", 1: "numeric"},
        scale_window="kNN",
        k_neighbours=k,
        scale="range",
    ).fit(data)

    expected_h_ratio = knn_bandwidth(data[:, 0], k=k)
    expected_h_numeric = knn_bandwidth(data[:, 1], k=k)

    assert pytest.approx(gs._h_ratio[0], rel=1e-12) == expected_h_ratio
    assert pytest.approx(gs._h_numeric[0], rel=1e-12) == expected_h_numeric

    d_AB = gs.distance(data[0], data[1])
    assert pytest.approx(d_AB, abs=1e-12) == 0.0

    d_AC = gs.distance(data[0], data[2])
    d_BC = gs.distance(data[1], data[2])
    assert 0.0 < d_AC <= 1.0
    assert 0.0 < d_BC <= 1.0

    assert pytest.approx(d_AC, rel=1e-12) == gs.distance(data[2], data[0])
    assert pytest.approx(d_BC, rel=1e-12) == gs.distance(data[2], data[1])
