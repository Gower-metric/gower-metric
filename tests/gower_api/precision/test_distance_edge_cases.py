"""Tests for distance component edge cases — ordinal metadata, ratio zero range, podani fallback."""

import warnings

import numpy as np
import pandas as pd
import pytest

from gower_metric import Config, Gower
from tests.gower_api.precision.conftest import BaseTest


class TestCategoricalOrdinalEdgeCases(BaseTest):
    def test_podani_fallback_to_kaufman(self) -> None:
        """Podani denominator <= 0 triggers fallback to kaufman."""
        data = np.array(
            [
                [1.0, "a"],
                [2.0, "a"],
                [3.0, "a"],
                [4.0, "b"],
            ],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "numeric", 1: "categorical_ordinal"},
            categorical_ordinal_values_order={1: ["a", "b"]},
            categorical_ordinal_calculation_type="podani",
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        with pytest.warns(UserWarning, match=r"Podani denominator"):
            dist = gower(data[0], data[3])
        assert 0.0 <= self.dtype(dist) <= 1.0

    def test_podani_fallback_pandas(self) -> None:
        """Same podani fallback test with pandas DataFrame."""
        data = pd.DataFrame(
            {
                "value": [1.0, 2.0, 3.0, 4.0],
                "grade": ["a", "a", "a", "b"],
            },
        )
        cfg = Config(
            feature_types={"value": "numeric", "grade": "categorical_ordinal"},
            categorical_ordinal_values_order={"grade": ["a", "b"]},
            categorical_ordinal_calculation_type="podani",
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        with pytest.warns(UserWarning, match=r"Podani denominator"):
            dist = gower(data.iloc[0], data.iloc[3])
        assert 0.0 <= self.dtype(dist) <= 1.0

    def test_kaufman_zero_denom_returns_zero_dist(self) -> None:
        """All ordinal values the same → denom=0 → dist=0."""
        data = np.array(
            [
                [1.0, "a"],
                [2.0, "a"],
            ],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "numeric", 1: "categorical_ordinal"},
            categorical_ordinal_values_order={1: ["a"]},
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data[0], data[1])
        assert 0.0 <= self.dtype(dist) <= 1.0

    def test_kaufman_zero_denom_pandas(self) -> None:
        data = pd.DataFrame({"val": [1.0, 2.0], "grade": ["a", "a"]})
        cfg = Config(
            feature_types={"val": "numeric", "grade": "categorical_ordinal"},
            categorical_ordinal_values_order={"grade": ["a"]},
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data.iloc[0], data.iloc[1])
        assert 0.0 <= self.dtype(dist) <= 1.0


class TestRatioScaleZeroRange(BaseTest):
    def test_zero_range_ratio_produces_zero_diff(self) -> None:
        """Constant ratio column → range=0 → diff should be 0."""
        data = np.array(
            [
                [5.0, "A"],
                [5.0, "B"],
                [5.0, "A"],
            ],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "ratio_scale_interval", 1: "categorical_nominal"},
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data[0], data[1])
        assert 0.0 <= self.dtype(dist) <= 1.0

    def test_zero_range_ratio_pandas(self) -> None:
        data = pd.DataFrame({"ratio": [5.0, 5.0, 5.0], "cat": ["A", "B", "A"]})
        cfg = Config(
            feature_types={
                "ratio": "ratio_scale_interval",
                "cat": "categorical_nominal",
            },
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data.iloc[0], data.iloc[1])
        assert 0.0 <= self.dtype(dist) <= 1.0


class TestMissingStrategyIntegration(BaseTest):
    def test_max_dist_strategy(self) -> None:
        data = np.array(
            [
                [1.0, "A"],
                [2.0, "B"],
                [np.nan, "A"],
            ],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "numeric", 1: "categorical_nominal"},
            missing_strategy="max_dist",
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data[0], data[2])
        assert self.dtype(dist) > 0

    def test_max_dist_strategy_pandas(self) -> None:
        data = pd.DataFrame({"val": [1.0, 2.0, np.nan], "cat": ["A", "B", "A"]})
        cfg = Config(
            feature_types={"val": "numeric", "cat": "categorical_nominal"},
            missing_strategy="max_dist",
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data.iloc[0], data.iloc[2])
        assert self.dtype(dist) > 0

    def test_raise_error_strategy(self) -> None:
        data = np.array(
            [
                [1.0, "A"],
                [2.0, "B"],
            ],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "numeric", 1: "categorical_nominal"},
            missing_strategy="raise_error",
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data[0], data[1])
        assert 0.0 <= self.dtype(dist) <= 1.0


class TestMaxDistBounded(BaseTest):
    """Max_dist strategy must produce distances in [0, 1]."""

    def test_max_dist_single_missing_feature(self) -> None:
        data = np.array(
            [[1.0, "A"], [2.0, np.nan]],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "numeric", 1: "categorical_nominal"},
            missing_strategy="max_dist",
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data[0], data[1])
        assert 0.0 <= self.dtype(dist) <= 1.0

    def test_max_dist_all_missing_one_side(self) -> None:
        data = np.array(
            [[1.0, "A", 10.0], [np.nan, np.nan, np.nan]],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "numeric", 1: "categorical_nominal", 2: "numeric"},
            missing_strategy="max_dist",
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data[0], data[1])
        assert 0.0 <= self.dtype(dist) <= 1.0
        assert dist == pytest.approx(1.0, rel=1e-6)

    def test_max_dist_partial_missing_bounded(self) -> None:
        """Multiple features, some present with nonzero dist, some missing.

        Previously, the denominator excluded missing pairs, making dist > 1.0.
        """
        data = np.array(
            [[0.0, "A", 5.0], [10.0, np.nan, np.nan]],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "numeric", 1: "categorical_nominal", 2: "numeric"},
            missing_strategy="max_dist",
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data[0], data[1])
        assert 0.0 <= self.dtype(dist) <= 1.0


class TestRaiseErrorMessage(BaseTest):
    """Raise_error strategy provides descriptive error message."""

    def test_raise_error_message_content(self) -> None:
        data = np.array(
            [[1.0, "A"], [np.nan, "B"]],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "numeric", 1: "categorical_nominal"},
            missing_strategy="raise_error",
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)
        with pytest.raises(ValueError, match="Missing values detected"):
            gower(data[0], data[1])


class TestPodaniNaN(BaseTest):
    """Podani calculation with NaN should not produce RuntimeWarning."""

    def test_podani_with_missing_no_warning(self) -> None:
        data = np.array(
            [["low", "A"], ["high", "B"], [np.nan, "A"]],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "categorical_ordinal", 1: "categorical_nominal"},
            categorical_ordinal_values_order={0: ["low", "medium", "high"]},
            categorical_ordinal_calculation_type="podani",
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dist = gower(data[0], data[2])
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) == 0, (
                f"Unexpected RuntimeWarning(s): {[str(x.message) for x in runtime_warnings]}"
            )

        assert dist == pytest.approx(0.0)


class TestPodaniFallbackWarning(BaseTest):
    """Podani fallback to Kaufman should emit a warning."""

    def test_podani_fallback_emits_warning(self) -> None:
        """When all ordinal values are the same, podani_denom <= 0 and Kaufman fallback triggers."""
        data = np.array(
            [["low"], ["low"]],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "categorical_ordinal"},
            categorical_ordinal_values_order={0: ["low"]},
            categorical_ordinal_calculation_type="podani",
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dist = gower(data[0], data[1])
            podani_warnings = [
                x
                for x in w
                if issubclass(x.category, UserWarning)
                and "Podani denominator" in str(x.message)
            ]
            assert len(podani_warnings) >= 1
        assert dist == pytest.approx(0.0)
