import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import pairwise_distances

from gower_metric import Config, Gower
from gower_metric.utils.auxiliary import all_ones_off_diagonal
from tests.gower_api.precision.conftest import BaseTest


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
        def d(x, y):
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

        def d(x, y):
            return abs(x - y) / r_max_min

        assert pytest.approx(gower(raw.iloc[0], raw.iloc[1]), rel=1e-6) == d(0, 2)
        assert pytest.approx(gower(raw.iloc[0], raw.iloc[3]), rel=1e-6) == d(0, 7)

    def test_conditional_distances_clip(self) -> None:
        raw = np.array(
            [
                ["A", "X", 0.0],
                ["A", "Y", 2.0],
                ["B", "X", 5.0],
                ["B", "Y", 7.0],
            ],
            dtype=object,
        )
        f_types: dict[int | str, str] = {
            0: "categorical_nominal",
            1: "categorical_ordinal",
            2: "numeric",
        }

        categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
            1: ["X", "Y"],
        }

        cfg = Config(
            feature_types=f_types,
            conditional_distances=True,
            categorical_ordinal_values_order=categorical_ordinal_values_order,
            data_type=self.dtype,
        )

        gower = Gower(cfg).fit(raw)

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
            dtype=self.dtype,
        )

        for i in range(raw.shape[0]):
            for j in range(raw.shape[0]):
                dist = gower(raw[i], raw[j])

                assert pytest.approx(dist, rel=1e-6) == expected[i, j]

    def test_value_error_on_no_numerical_features(self) -> None:
        f_types: dict[int | str, str] = {
            0: "binary_asymmetric",
            1: "categorical_nominal",
            2: "categorical_ordinal",
        }

        categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
            2: ["X", "Y"],
        }

        with pytest.raises(ValueError):
            Config(
                feature_types=f_types,
                conditional_distances=True,
                categorical_ordinal_values_order=categorical_ordinal_values_order,
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
                ["A", "X", 0.0],
                ["B", "Y", 2.0],
                ["C", "Z", 0.0],
            ],
            dtype=object,
        )
        f_types: dict[int | str, str] = {
            0: "categorical_nominal",
            1: "categorical_ordinal",
            2: "numeric",
        }

        categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
            1: ["X", "Y", "Z"],
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
