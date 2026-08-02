# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

from gower_metric import Config, Gower
from gower_metric.utils.discretization_types import knn, silverman
from tests.gower_api.precision.base import BaseTest


class TestComputeBandwidth(BaseTest):
    def test_ratio_scale_knn_window_no_error(self) -> None:
        rng = np.random.default_rng(seed=42)
        data = rng.normal(size=(60, 2))

        cfg = Config(
            feature_types={0: "ratio_scale_interval", 1: "numeric"},
            discretization=knn.NAME,
            scale_method="range",
            data_type=self.dtype,
        )
        gs_knn = Gower(cfg).fit(data)

        assert isinstance(gs_knn._h_ratio, np.ndarray)
        assert (np.asarray(gs_knn._h_ratio) > 0).all()
        assert (np.asarray(gs_knn._h_numeric) > 0).all()

    def test_ratio_scale_knn_window_pandas(self) -> None:
        rng = np.random.default_rng(seed=42)
        arr = rng.normal(size=(60, 2))
        data = pd.DataFrame(arr, columns=["ratio_col", "num_col"])

        cfg = Config(
            feature_types={"ratio_col": "ratio_scale_interval", "num_col": "numeric"},
            discretization=knn.NAME,
            scale_method="range",
            data_type=self.dtype,
        )
        gs_knn = Gower(cfg).fit(data)

        assert isinstance(gs_knn._h_ratio, np.ndarray)
        assert (np.asarray(gs_knn._h_ratio) > 0).all()
        assert (np.asarray(gs_knn._h_numeric) > 0).all()

    def test_ratio_scale_discretization_multi(self) -> None:
        rng = np.random.default_rng(seed=123)
        data = rng.normal(size=(80, 2))

        cfg = Config(
            feature_types={0: "ratio_scale_interval", 1: "numeric"},
            discretization=silverman.NAME,
            scale_method="range",
            data_type=self.dtype,
        )
        gs_discret = Gower(cfg).fit(data)

        assert isinstance(gs_discret._h_ratio, np.ndarray)
        assert gs_discret._h_ratio.shape == (1,)
        assert isinstance(gs_discret._h_numeric, np.ndarray)
        assert gs_discret._h_numeric.shape == (1,)

        manual_h_ratio = silverman.bandwidth(data[:, 0])
        manual_h_numeric = silverman.bandwidth(data[:, 1])

        assert pytest.approx(gs_discret._h_ratio[0], rel=1e-12) == manual_h_ratio
        assert pytest.approx(gs_discret._h_numeric[0], rel=1e-12) == manual_h_numeric

    def test_knn_bandwidth_values_and_effect(self) -> None:
        data = np.array(
            [
                [0.00, 0.00],
                [0.10, 0.05],
                [2.00, 1.00],
            ],
            dtype=self.dtype,
        )

        k = 1
        cfg = Config(
            feature_types={0: "ratio_scale_interval", 1: "numeric"},
            discretization=knn.NAME,
            k_neighbors=k,
            scale_method="range",
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)

        expected_h_ratio = knn.bandwidth(data[:, 0], k=k)
        expected_h_numeric = knn.bandwidth(data[:, 1], k=k)

        assert pytest.approx(gower._h_ratio[0], rel=1e-12) == expected_h_ratio
        assert pytest.approx(gower._h_numeric[0], rel=1e-12) == expected_h_numeric

        d_AB = gower(data[0], data[1])
        assert pytest.approx(d_AB, abs=1e-12) == 0.0

        d_AC = gower(data[0], data[2])
        d_BC = gower(data[1], data[2])
        assert 0.0 < d_AC <= 1.0
        assert 0.0 < d_BC <= 1.0

        assert pytest.approx(d_AC, rel=1e-12) == gower(data[2], data[0])
        assert pytest.approx(d_BC, rel=1e-12) == gower(data[2], data[1])
