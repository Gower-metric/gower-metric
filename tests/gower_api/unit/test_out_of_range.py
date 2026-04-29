"""Tests for Config.out_of_range parameter — clip, warning, error."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gower_metric import Config, Gower
from gower_metric.core.config import FeatureType, OutOfRangeStrategy

NUMERIC_TYPES = ["numeric", "ratio_scale_interval"]


def _config(
    feat_type: FeatureType,
    strategy: OutOfRangeStrategy,
    **extra: Any,
) -> Config:
    """Build a Config with one numeric-like + one categorical column."""
    kw: dict[str, Any] = {
        "feature_types": {0: feat_type, 1: "categorical_nominal"},
        "out_of_range": strategy,
    }
    kw.update(extra)
    return Config(**kw)


class TestOutOfRangeTransform:
    """out_of_range behaviour during transform()."""

    @pytest.fixture(autouse=True, params=NUMERIC_TYPES)
    def setup_feat_type(self, request: pytest.FixtureRequest) -> None:
        self.feat_type: FeatureType = request.param

    def _label(self) -> str:
        return "numeric" if self.feat_type == "numeric" else "ratio_scale"

    def test_warning_emitted_on_oor(self) -> None:
        train = np.array([[1.0, "A"], [5.0, "B"]], dtype=object)
        test = np.array([[6.0, "A"]], dtype=object)
        gower = Gower(_config(self.feat_type, "warning")).fit(train)

        with pytest.warns(UserWarning, match=r"Out-of-range values detected"):
            gower.transform(test)

    def test_error_raised_on_oor(self) -> None:
        train = np.array([[1.0, "A"], [5.0, "B"]], dtype=object)
        test = np.array([[6.0, "A"]], dtype=object)
        gower = Gower(_config(self.feat_type, "error")).fit(train)

        with pytest.raises(ValueError, match=r"Out-of-range values detected"):
            gower.transform(test)

    def test_clip_silent_on_oor(self) -> None:
        train = np.array([[1.0, "A"], [5.0, "B"]], dtype=object)
        test = np.array([[6.0, "A"]], dtype=object)
        gower = Gower(_config(self.feat_type, "clip")).fit(train)

        result = gower.transform(test)
        assert result.shape == (1, 2)

    def test_in_range_no_warning(self) -> None:
        train = np.array([[1.0, "A"], [5.0, "B"]], dtype=object)
        test = np.array([[3.0, "A"]], dtype=object)
        gower = Gower(_config(self.feat_type, "warning")).fit(train)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gower.transform(test)

    def test_below_min_triggers_oor(self) -> None:
        train = np.array([[2.0, "A"], [5.0, "B"]], dtype=object)
        test = np.array([[0.0, "A"]], dtype=object)
        gower = Gower(_config(self.feat_type, "error")).fit(train)

        with pytest.raises(ValueError, match=r"Out-of-range"):
            gower.transform(test)

    def test_nan_does_not_trigger_oor(self) -> None:
        train = np.array([[1.0, "A"], [5.0, "B"]], dtype=object)
        test = np.array([[np.nan, "A"]], dtype=object)
        gower = Gower(_config(self.feat_type, "error")).fit(train)

        result = gower.transform(test)
        assert result.shape == (1, 2)

    def test_warning_message_contains_column_details(self) -> None:
        train = np.array([[1.0, "A"], [5.0, "B"]], dtype=object)
        test = np.array([[10.0, "A"]], dtype=object)
        gower = Gower(_config(self.feat_type, "warning")).fit(train)

        with pytest.warns(UserWarning, match=rf"{self._label()} column 0"):
            gower.transform(test)

    def test_pandas_input(self) -> None:
        train = pd.DataFrame({"val": [1.0, 5.0], "cat": ["A", "B"]})
        test = pd.DataFrame({"val": [6.0], "cat": ["A"]})
        cfg = Config(
            feature_types={"val": self.feat_type, "cat": "categorical_nominal"},
            out_of_range="warning",
        )
        gower = Gower(cfg).fit(train)

        with pytest.warns(UserWarning, match=r"Out-of-range"):
            gower.transform(test)


class TestOutOfRangeCall:
    """out_of_range behaviour during __call__()."""

    def test_warning_on_call(self) -> None:
        train = np.array([[1.0], [5.0]], dtype=object)
        cfg = Config(feature_types={0: "numeric"}, out_of_range="warning")
        gower = Gower(cfg).fit(train)

        with pytest.warns(UserWarning, match=r"Out-of-range"):
            gower(np.array([6.0], dtype=object), np.array([3.0], dtype=object))

    def test_error_on_call(self) -> None:
        train = np.array([[1.0], [5.0]], dtype=object)
        cfg = Config(feature_types={0: "numeric"}, out_of_range="error")
        gower = Gower(cfg).fit(train)

        with pytest.raises(ValueError, match=r"Out-of-range"):
            gower(np.array([6.0], dtype=object), np.array([3.0], dtype=object))

    def test_clip_on_call(self) -> None:
        train = np.array([[1.0], [5.0]], dtype=object)
        cfg = Config(feature_types={0: "numeric"}, out_of_range="clip")
        gower = Gower(cfg).fit(train)

        result = gower(np.array([6.0], dtype=object), np.array([3.0], dtype=object))
        assert not np.isnan(result)


class TestOutOfRangeMatrix:
    """out_of_range behaviour during matrix()."""

    def test_warning_on_matrix(self) -> None:
        train = np.array([[1.0], [5.0]], dtype=object)
        test = np.array([[1.0], [5.0], [10.0]], dtype=object)
        cfg = Config(feature_types={0: "numeric"}, out_of_range="warning")
        gower = Gower(cfg).fit(train)

        with pytest.warns(UserWarning, match=r"Out-of-range"):
            gower.matrix(test)

    def test_error_on_matrix(self) -> None:
        train = np.array([[1.0], [5.0]], dtype=object)
        test = np.array([[1.0], [5.0], [10.0]], dtype=object)
        cfg = Config(feature_types={0: "numeric"}, out_of_range="error")
        gower = Gower(cfg).fit(train)

        with pytest.raises(ValueError, match=r"Out-of-range"):
            gower.matrix(test)

    def test_clip_on_matrix(self) -> None:
        train = np.array([[1.0], [5.0]], dtype=object)
        test = np.array([[1.0], [5.0], [10.0]], dtype=object)
        cfg = Config(feature_types={0: "numeric"}, out_of_range="clip")
        gower = Gower(cfg).fit(train)

        result = gower.matrix(test)
        assert result.shape == (3, 3)
