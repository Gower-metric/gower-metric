"""Tests for unique transform branches — NaN passthrough, degenerate columns, dtype variants."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from gower_metric import Config, Gower

BINARY_TYPES = ["binary_asymmetric", "binary_symmetric"]
CATEGORICAL_TYPES = ["categorical_nominal", "categorical_ordinal"]


def _binary_config(
    binary_type: str,
    feature_types: dict[int | str, str],
    **extra: Any,
) -> Config:
    """Build a Config for a binary + numeric column pair."""
    kw: dict[str, Any] = {
        "feature_types": feature_types,
        f"handle_unseen_{binary_type}": "missing",
        "out_of_range": "clip",
    }
    kw.update(extra)
    return Config(**kw)


def _categorical_config(cat_type: str, **extra: Any) -> Config:
    """Build a Config for a single categorical + numeric column."""
    kw: dict[str, Any] = {
        "feature_types": {0: cat_type, 1: "numeric"},
        f"handle_unseen_{cat_type}": "warning",
        "out_of_range": "clip",
    }
    if cat_type == "categorical_ordinal":
        kw["categorical_ordinal_values_order"] = extra.pop(
            "values_order",
            {0: ["low", "medium", "high"]},
        )
    kw.update(extra)
    return Config(**kw)


def _categorical_config_pandas(cat_type: str) -> Config:
    """Build a Config for a pandas categorical + numeric column pair."""
    kw: dict[str, Any] = {
        "feature_types": {"cat": cat_type, "val": "numeric"},
        f"handle_unseen_{cat_type}": "warning",
        "out_of_range": "clip",
    }
    if cat_type == "categorical_ordinal":
        kw["categorical_ordinal_values_order"] = {"cat": ["low", "medium", "high"]}
    return Config(**kw)


class TestBinaryNaNPassthrough:
    """NaN values in binary columns pass through as NaN."""

    @pytest.fixture(autouse=True, params=BINARY_TYPES)
    def setup_binary_type(self, request: pytest.FixtureRequest) -> None:
        self.binary_type: str = request.param

    def test_nan_passthrough_numpy(self) -> None:
        train = np.array([[0, 1.0], [1, 2.0]], dtype=object)
        test = np.array([[np.nan, 3.0]], dtype=object)
        cfg = _binary_config(self.binary_type, {0: self.binary_type, 1: "numeric"})
        result = Gower(cfg).fit(train).transform(test)
        assert np.isnan(float(result[0][0]))

    def test_nan_passthrough_pandas(self) -> None:
        train = pd.DataFrame({"flag": [0, 1], "val": [1.0, 2.0]})
        test = pd.DataFrame({"flag": [np.nan], "val": [3.0]})
        cfg = _binary_config(
            self.binary_type,
            {"flag": self.binary_type, "val": "numeric"},
        )
        result = Gower(cfg).fit(train).transform(test)
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(float(result.iloc[0, 0]))


class TestDegenerateBinaryColumn:
    """Binary column with all NaN values should still fit."""

    @pytest.fixture(autouse=True, params=BINARY_TYPES)
    def setup_binary_type(self, request: pytest.FixtureRequest) -> None:
        self.binary_type: str = request.param

    def test_all_nan_fit_numpy(self) -> None:
        train = np.array([[np.nan, 1.0], [np.nan, 2.0]], dtype=object)
        cfg = _binary_config(self.binary_type, {0: self.binary_type, 1: "numeric"})
        result = Gower(cfg).fit(train).transform(train)
        assert result.shape == (2, 2)

    def test_all_nan_fit_pandas(self) -> None:
        train = pd.DataFrame({"flag": [np.nan, np.nan], "val": [1.0, 2.0]})
        cfg = _binary_config(
            self.binary_type,
            {"flag": self.binary_type, "val": "numeric"},
        )
        result = Gower(cfg).fit(train).transform(train)
        assert result.shape == (2, 2)


class TestCategoricalNaNTransform:
    """NaN in categorical columns produces NaN output."""

    @pytest.fixture(autouse=True, params=CATEGORICAL_TYPES)
    def setup_cat_type(self, request: pytest.FixtureRequest) -> None:
        self.cat_type: str = request.param

    def _train_data(self) -> np.ndarray:
        if self.cat_type == "categorical_ordinal":
            return np.array([["low", 1.0], ["high", 2.0]], dtype=object)
        return np.array([["A", 1.0], ["B", 2.0]], dtype=object)

    def test_nan_transform_produces_nan(self) -> None:
        test = np.array([[np.nan, 3.0]], dtype=object)
        cfg = _categorical_config(self.cat_type)
        result = Gower(cfg).fit(self._train_data()).transform(test)
        assert np.isnan(float(result[0][0]))

    def test_no_unseen_warning_not_triggered(self) -> None:
        """Warning strategy configured but all values seen — no warning fired."""
        train = self._train_data()
        extra = (
            {"values_order": {0: ["low", "high"]}}
            if self.cat_type == "categorical_ordinal"
            else {}
        )
        cfg = _categorical_config(self.cat_type, **extra)
        result = Gower(cfg).fit(train).transform(train)
        assert not np.isnan(float(result[0][0]))

    def test_nan_transform_pandas(self) -> None:
        train = (
            pd.DataFrame({"cat": ["low", "high"], "val": [1.0, 2.0]})
            if self.cat_type == "categorical_ordinal"
            else pd.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]})
        )
        test = pd.DataFrame({"cat": [np.nan], "val": [3.0]})
        cfg = _categorical_config_pandas(self.cat_type)
        result = Gower(cfg).fit(train).transform(test)
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(float(result.iloc[0, 0]))


class TestTransformDtypeVariants:
    """Transform output dtype matches configured data_type."""

    @pytest.fixture(autouse=True, params=[np.float32, np.float64])
    def setup_dtype(self, request: pytest.FixtureRequest) -> None:
        self.dtype = request.param

    def test_output_dtype_numpy(self) -> None:
        train = np.array([["A", 1.0], ["B", 2.0]], dtype=object)
        cfg = Config(
            feature_types={0: "categorical_nominal", 1: "numeric"},
            data_type=self.dtype,
        )
        result = Gower(cfg).fit(train).transform(train)
        assert result.dtype == self.dtype

    def test_output_dtype_pandas(self) -> None:
        train = pd.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]})
        cfg = Config(
            feature_types={"cat": "categorical_nominal", "val": "numeric"},
            data_type=self.dtype,
        )
        result = Gower(cfg).fit(train).transform(train)
        assert isinstance(result, pd.DataFrame)
        assert (result.dtypes == self.dtype).all()
