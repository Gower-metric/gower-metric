"""Tests for transform branch coverage — explicit order violations, categorical unseen branches."""

import numpy as np
import pandas as pd
import pytest

from gower_metric import Config, Gower


class TestBinaryExplicitOrderViolation:
    def test_asymmetric_explicit_order_violation_in_transform(self) -> None:
        """Value not in explicit order during transform raises ValueError."""
        train = np.array([[0, 1.0], [1, 2.0]])
        test = np.array([[2, 3.0]])  # value 2 not in [0, 1]
        cfg = Config(
            feature_types={0: "binary_asymmetric", 1: "numeric"},
            binary_asymmetric_value_order={0: [0, 1]},
        )
        gower = Gower(cfg).fit(train)
        with pytest.raises(ValueError, match="violates binary_asymmetric_value_order"):
            gower.transform(test)

    def test_symmetric_explicit_order_violation_in_transform(self) -> None:
        """Value not in explicit order during transform raises ValueError."""
        train = np.array([[0, 1.0], [1, 2.0]])
        test = np.array([[2, 3.0]])  # value 2 not in [0, 1]
        cfg = Config(
            feature_types={0: "binary_symmetric", 1: "numeric"},
            binary_symmetric_value_order={0: [0, 1]},
        )
        gower = Gower(cfg).fit(train)
        with pytest.raises(ValueError, match="violates binary_symmetric_value_order"):
            gower.transform(test)

    def test_asymmetric_explicit_order_violation_pandas(self) -> None:
        """Same as above using pandas DataFrame."""
        train = pd.DataFrame({"flag": [0, 1], "value": [1.0, 2.0]})
        test = pd.DataFrame({"flag": [2], "value": [3.0]})
        cfg = Config(
            feature_types={"flag": "binary_asymmetric", "value": "numeric"},
            binary_asymmetric_value_order={"flag": [0, 1]},
        )
        gower = Gower(cfg).fit(train)
        with pytest.raises(ValueError, match="violates binary_asymmetric_value_order"):
            gower.transform(test)


class TestCategoricalUnseenWarningBranch:
    def test_nominal_unseen_warning(self) -> None:
        """handle_unseen='warning' for categorical nominal maps unseen to NaN."""
        train = np.array([["A", 1.0], ["B", 2.0]], dtype=object)
        test = np.array([["C", 3.0]], dtype=object)
        cfg = Config(
            feature_types={0: "categorical_nominal", 1: "numeric"},
            handle_unseen_categorical_nominal="warning",
        )
        gower = Gower(cfg).fit(train)
        with pytest.warns(UserWarning, match="Unseen values"):
            result = gower.transform(test)
        assert np.isnan(result[0, 0]) or np.isnan(float(result[0][0]))

    def test_ordinal_unseen_warning(self) -> None:
        """handle_unseen='warning' for categorical ordinal maps unseen to NaN."""
        train = np.array([["low", 1.0], ["high", 2.0]], dtype=object)
        test = np.array([["unseen", 3.0]], dtype=object)
        cfg = Config(
            feature_types={0: "categorical_ordinal", 1: "numeric"},
            categorical_ordinal_values_order={0: ["low", "medium", "high"]},
            handle_unseen_categorical_ordinal="warning",
        )
        gower = Gower(cfg).fit(train)
        with pytest.warns(UserWarning, match="Unseen values"):
            result = gower.transform(test)
        assert np.isnan(float(result[0][0]))

    def test_nominal_unseen_warning_pandas(self) -> None:
        """Same unseen warning test with pandas."""
        train = pd.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]})
        test = pd.DataFrame({"cat": ["C"], "val": [3.0]})
        cfg = Config(
            feature_types={"cat": "categorical_nominal", "val": "numeric"},
            handle_unseen_categorical_nominal="warning",
        )
        gower = Gower(cfg).fit(train)
        with pytest.warns(UserWarning, match="Unseen values"):
            result = gower.transform(test)
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(float(result.iloc[0, 0]))


class TestTransformWithNaN:
    def test_binary_asymmetric_nan_passthrough(self) -> None:
        """NaN values in binary asymmetric columns pass through as NaN."""
        train = np.array([[0, 1.0], [1, 2.0]], dtype=object)
        test = np.array([[np.nan, 3.0]], dtype=object)
        cfg = Config(
            feature_types={0: "binary_asymmetric", 1: "numeric"},
            handle_unseen_binary_asymmetric="missing",
        )
        gower = Gower(cfg).fit(train)
        result = gower.transform(test)
        assert np.isnan(float(result[0][0]))

    def test_binary_symmetric_nan_passthrough(self) -> None:
        """NaN values in binary symmetric columns pass through as NaN."""
        train = np.array([[0, 1.0], [1, 2.0]], dtype=object)
        test = np.array([[np.nan, 3.0]], dtype=object)
        cfg = Config(
            feature_types={0: "binary_symmetric", 1: "numeric"},
            handle_unseen_binary_symmetric="missing",
        )
        gower = Gower(cfg).fit(train)
        result = gower.transform(test)
        assert np.isnan(float(result[0][0]))

    def test_binary_nan_passthrough_pandas(self) -> None:
        """NaN passthrough with pandas DataFrame."""
        train = pd.DataFrame({"flag": [0, 1], "val": [1.0, 2.0]})
        test = pd.DataFrame({"flag": [np.nan], "val": [3.0]})
        cfg = Config(
            feature_types={"flag": "binary_asymmetric", "val": "numeric"},
            handle_unseen_binary_asymmetric="missing",
        )
        gower = Gower(cfg).fit(train)
        result = gower.transform(test)
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(float(result.iloc[0, 0]))


class TestDegenerateBinaryColumn:
    def test_all_nan_binary_column_fit(self) -> None:
        """Binary column with all NaN values should still fit (empty mapping)."""
        train = np.array([[np.nan, 1.0], [np.nan, 2.0]], dtype=object)
        cfg = Config(
            feature_types={0: "binary_symmetric", 1: "numeric"},
            handle_unseen_binary_symmetric="missing",
        )
        gower = Gower(cfg).fit(train)
        result = gower.transform(train)
        assert result.shape == (2, 2)

    def test_all_nan_binary_column_fit_pandas(self) -> None:
        """Same with pandas."""
        train = pd.DataFrame({"flag": [np.nan, np.nan], "val": [1.0, 2.0]})
        cfg = Config(
            feature_types={"flag": "binary_symmetric", "val": "numeric"},
            handle_unseen_binary_symmetric="missing",
        )
        gower = Gower(cfg).fit(train)
        result = gower.transform(train)
        assert result.shape == (2, 2)


class TestAllNaNCategoricalColumn:
    def test_all_nan_nominal_transform(self) -> None:
        """All-NaN nominal column should produce all-NaN output."""
        train = np.array([["A", 1.0], ["B", 2.0]], dtype=object)
        test = np.array([[np.nan, 3.0]], dtype=object)
        cfg = Config(
            feature_types={0: "categorical_nominal", 1: "numeric"},
            handle_unseen_categorical_nominal="warning",
        )
        gower = Gower(cfg).fit(train)
        result = gower.transform(test)
        assert np.isnan(float(result[0][0]))

    def test_all_nan_ordinal_transform(self) -> None:
        """All-NaN ordinal column should produce all-NaN output."""
        train = np.array([["low", 1.0], ["high", 2.0]], dtype=object)
        test = np.array([[np.nan, 3.0]], dtype=object)
        cfg = Config(
            feature_types={0: "categorical_ordinal", 1: "numeric"},
            categorical_ordinal_values_order={0: ["low", "medium", "high"]},
            handle_unseen_categorical_ordinal="warning",
        )
        gower = Gower(cfg).fit(train)
        result = gower.transform(test)
        assert np.isnan(float(result[0][0]))

    def test_nominal_no_unseen_with_warning_strategy(self) -> None:
        """All values seen → warning handler branch not triggered."""
        train = np.array([["A", 1.0], ["B", 2.0]], dtype=object)
        cfg = Config(
            feature_types={0: "categorical_nominal", 1: "numeric"},
            handle_unseen_categorical_nominal="warning",
        )
        gower = Gower(cfg).fit(train)
        result = gower.transform(train)
        assert not np.isnan(float(result[0][0]))

    def test_ordinal_no_unseen_with_warning_strategy(self) -> None:
        """All values seen → warning handler branch not triggered."""
        train = np.array([["low", 1.0], ["high", 2.0]], dtype=object)
        cfg = Config(
            feature_types={0: "categorical_ordinal", 1: "numeric"},
            categorical_ordinal_values_order={0: ["low", "high"]},
            handle_unseen_categorical_ordinal="warning",
        )
        gower = Gower(cfg).fit(train)
        result = gower.transform(train)
        assert not np.isnan(float(result[0][0]))

    def test_all_nan_nominal_pandas(self) -> None:
        """All-NaN nominal column with pandas."""
        train = pd.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]})
        test = pd.DataFrame({"cat": [np.nan], "val": [3.0]})
        cfg = Config(
            feature_types={"cat": "categorical_nominal", "val": "numeric"},
            handle_unseen_categorical_nominal="warning",
        )
        gower = Gower(cfg).fit(train)
        result = gower.transform(test)
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(float(result.iloc[0, 0]))


class TestTransformDtypeVariants:
    def test_transform_float64(self) -> None:
        """Transform output respects float64 data_type."""
        train = np.array([["A", 1.0], ["B", 2.0]], dtype=object)
        cfg = Config(
            feature_types={0: "categorical_nominal", 1: "numeric"},
            data_type=np.float64,
        )
        gower = Gower(cfg).fit(train)
        result = gower.transform(train)
        assert result.dtype == np.float64

    def test_transform_float64_pandas(self) -> None:
        """Transform output respects float64 data_type with pandas input."""
        train = pd.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]})
        cfg = Config(
            feature_types={"cat": "categorical_nominal", "val": "numeric"},
            data_type=np.float64,
        )
        gower = Gower(cfg).fit(train)
        result = gower.transform(train)
        assert isinstance(result, pd.DataFrame)
        assert (result.dtypes == np.float64).all()
