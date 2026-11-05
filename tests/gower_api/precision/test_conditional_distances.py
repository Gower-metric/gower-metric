# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import pairwise_distances

from gower_metric import Config, Gower
from gower_metric.utils.auxiliary import all_ones_off_diagonal
from tests.gower_api.precision.base import BaseTest


class TestConditionalDistances(BaseTest):
    def test_conditional_distances(self) -> None:
        raw = np.array(
            [
                ["A", 0.0],
                ["A", 2.0],
                ["B", 5.0],
                ["B", 7.0],
            ],
            dtype=object,
        )
        f_types: dict[int | str, str] = {0: "categorical_nominal", 1: "numeric"}

        cfg = Config(
            feature_types=f_types,
            conditional_distances=True,
            data_type=self.dtype,
        )

        gower = Gower(cfg).fit(raw)

        # max - min
        r_max_min = 7 - 0

        # Manhattan distance, normalized by range
        def d(x: float, y: float) -> float:
            return abs(x - y) / r_max_min

        expected = np.array(
            [
                [d(0, 0), d(0, 2), d(0, 5), d(0, 7)],
                [d(2, 0), d(2, 2), d(2, 5), d(2, 7)],
                [d(5, 0), d(5, 2), d(5, 5), d(5, 7)],
                [d(7, 0), d(7, 2), d(7, 5), d(7, 7)],
            ],
            dtype=self.dtype,
        )

        for i in range(raw.shape[0]):
            for j in range(raw.shape[0]):
                dist = gower(raw[i], raw[j])

                assert pytest.approx(dist, rel=1e-6) == expected[i, j]

    def test_conditional_distances_pandas(self) -> None:
        raw = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B"],
                "value": [0.0, 2.0, 5.0, 7.0],
            },
        )
        cfg = Config(
            feature_types={"category": "categorical_nominal", "value": "numeric"},
            conditional_distances=True,
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(raw)

        r_max_min = 7 - 0

        def d(x: float, y: float) -> float:
            return abs(x - y) / r_max_min

        assert pytest.approx(gower(raw.iloc[0], raw.iloc[1]), rel=1e-6) == d(0, 2)
        assert pytest.approx(gower(raw.iloc[0], raw.iloc[3]), rel=1e-6) == d(0, 7)

    def test_conditional_distances_clip(self) -> None:
        """D'Orazio (2021, sec. 3): clip to 1 in step one, else step two only."""
        raw = np.array(
            [
                ["A", "P", "X", 0.0],
                ["A", "Q", "Y", 2.0],
                ["B", "Q", "X", 5.0],
                ["B", "P", "Y", 7.0],
            ],
            dtype=object,
        )
        f_types: dict[int | str, str] = {
            0: "categorical_nominal",
            1: "categorical_nominal",
            2: "categorical_ordinal",
            3: "numeric",
        }
        cfg = Config(
            feature_types=f_types,
            conditional_distances=True,
            categorical_ordinal_values_order={2: ["X", "Y"]},
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(raw)

        r = 7.0  # numeric range

        def step_two(ord_d: float, x: float, y: float) -> float:
            # ordinal X/Y: Kaufman over the observed span 1, so 0 or 1
            return (ord_d + abs(x - y) / r) / 2

        expected = np.array(
            [
                # both nominals differ on (0,2) and (1,3) -> clipped to 1
                [0.0, step_two(1, 0, 2), 1.0, step_two(1, 0, 7)],
                [step_two(1, 2, 0), 0.0, step_two(1, 2, 5), 1.0],
                [1.0, step_two(1, 5, 2), 0.0, step_two(1, 5, 7)],
                [step_two(1, 7, 0), 1.0, step_two(1, 7, 5), 0.0],
            ],
            dtype=self.dtype,
        )
        rel = 1e-3 if self.dtype == np.float16 else 1e-6
        for i in range(raw.shape[0]):
            for j in range(raw.shape[0]):
                dist = gower(raw[i], raw[j])
                assert pytest.approx(dist, rel=rel) == expected[i, j]

    def test_ordinal_alone_fills_the_second_step(self) -> None:
        """An ordinal variable is enough for step two (reading of D'Orazio 2021, p. 5).

        This configuration used to be rejected for lacking a numeric feature,
        on the old reading that ordinal variables belong to step one.
        """
        f_types: dict[int | str, str] = {
            0: "binary_asymmetric",
            1: "categorical_nominal",
            2: "categorical_ordinal",
        }
        Config(
            feature_types=f_types,
            conditional_distances=True,
            categorical_ordinal_values_order={2: ["X", "Y"]},
            data_type=self.dtype,
        )

    def test_value_error_on_no_categorical_features(self) -> None:
        f_types: dict[int | str, str] = {
            0: "numeric",
            1: "ratio_scale_interval",
        }

        with pytest.raises(ValueError):
            Config(
                feature_types=f_types,
                conditional_distances=True,
                data_type=self.dtype,
            )

    def test_value_error_on_too_small_threshold_coeff(self) -> None:
        f_types: dict[int | str, str] = {
            0: "numeric",
        }

        with pytest.raises(ValueError):
            Config(
                feature_types=f_types,
                conditional_distances=True,
                conditional_distances_threshold_coeff=0,
                data_type=self.dtype,
            )

    def test_conditional_distances_threshold_coeff(self) -> None:
        raw = np.array(
            [
                ["A", "P", "X", 0.0],
                ["B", "Q", "Y", 2.0],
                ["C", "R", "Z", 0.0],
            ],
            dtype=object,
        )
        f_types: dict[int | str, str] = {
            0: "categorical_nominal",
            1: "categorical_nominal",
            2: "categorical_ordinal",
            3: "numeric",
        }

        categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
            2: ["X", "Y", "Z"],
        }

        cfg = Config(
            feature_types=f_types,
            conditional_distances=True,
            categorical_ordinal_values_order=categorical_ordinal_values_order,
            data_type=self.dtype,
        )

        gower = Gower(cfg).fit(raw)

        pairwise_dist_result = pairwise_distances(raw, metric=gower)

        assert all_ones_off_diagonal(pairwise_dist_result)

        cfg2 = Config(
            feature_types=f_types,
            categorical_ordinal_values_order=categorical_ordinal_values_order,
            conditional_distances=True,
            conditional_distances_threshold_coeff=2,
            data_type=self.dtype,
        )

        gower = Gower(cfg2).fit(raw)

        pairwise_dist_result = pairwise_distances(raw, metric=gower)

        assert not all_ones_off_diagonal(pairwise_dist_result)
