"""Tests for binary_asymmetric_value_order parameter."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gower_metric import Config, Gower


class TestBinaryAsymmetricValueOrder:
    """Test explicit value ordering for binary_asymmetric features."""

    def test_explicit_order_complete_fit(self) -> None:
        """Test explicit ordering when training has both values.

        Training data: ['No', 'Yes', 'No']
        Test data: ['Yes', 'No'] -> 'Yes' -> 1.0, 'No' -> 0.0
        """
        X_train = np.array([["No"], ["Yes"], ["No"]], dtype=object)
        X_test = np.array([["Yes"], ["No"]], dtype=object)

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            binary_asymmetric_value_order={0: ["No", "Yes"]},
        )
        gower = Gower(cfg).fit(X_train)
        result = gower.transform(X_test)

        assert result[0, 0] == 1.0
        assert result[1, 0] == 0.0

    def test_explicit_order_degenerate_fit_sees_second_value(self) -> None:
        """Test that explicit ordering handles expected-but-not-yet-seen values.

        Training data: ['No', 'No']
        Test data: ['Yes'] -> 'Yes' -> 1.0
        """
        X_train = np.array([["No"], ["No"]], dtype=object)
        X_test = np.array([["Yes"]], dtype=object)

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            binary_asymmetric_value_order={0: ["No", "Yes"]},
        )
        gower = Gower(cfg).fit(X_train)
        result = gower.transform(X_test)

        assert result[0, 0] == 1.0

    def test_explicit_order_truly_unseen_value_always_raises(self) -> None:
        """Test that unseen values violating explicit order ALWAYS raise error.

        Training data: ['No', 'Yes']
        Test data: ['Maybe'] -> ValueError because 'Maybe' is not in order!
        """
        X_train = np.array([["No"], ["Yes"]], dtype=object)
        X_test = np.array([["Maybe"]], dtype=object)

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            binary_asymmetric_value_order={0: ["No", "Yes"]},
            handle_unseen_binary_asymmetric="missing",
        )
        gower = Gower(cfg).fit(X_train)

        with pytest.raises(
            ValueError,
            match=r"Value 'Maybe' in column 0 violates binary_asymmetric_value_order",
        ):
            gower.transform(X_test)

    def test_explicit_order_violation_ignores_strategy(self) -> None:
        """Test that explicit order violations raise error regardless of strategy.

        Training data: ['A', 'B']
        Test data: ['C'] -> ValueError because 'C' is not in order!
        """
        X_train = np.array([["A"], ["B"]], dtype=object)
        X_test = np.array([["C"]], dtype=object)

        for strategy in ["error", "warning", "missing"]:
            cfg = Config(
                feature_types={0: "binary_asymmetric"},
                binary_asymmetric_value_order={0: ["A", "B"]},
                handle_unseen_binary_asymmetric=strategy,
            )
            gower = Gower(cfg).fit(X_train)

            with pytest.raises(
                ValueError,
                match=r"violates binary_asymmetric_value_order",
            ):
                gower.transform(X_test)

    def test_explicit_order_training_has_unexpected_value(self) -> None:
        """Test that training data with unexpected values raises error.

        Training data: ['No', 'Maybe']
        """
        X_train = np.array([["No"], ["Maybe"]], dtype=object)

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            binary_asymmetric_value_order={0: ["No", "Yes"]},
        )

        with pytest.raises(
            ValueError,
            match=r"contains values not in binary_asymmetric_value_order.*Maybe",
        ):
            Gower(cfg).fit(X_train)

    def test_explicit_order_reversed_maintains_consistency(self) -> None:
        """Test that explicit ordering is not affected by data order.

        Run 1: fit on ['Yes', 'No']
        Run 2: fit on ['No', 'Yes']
        """
        X_train1 = np.array([["Yes"], ["No"]], dtype=object)
        cfg1 = Config(
            feature_types={0: "binary_asymmetric"},
            binary_asymmetric_value_order={0: ["No", "Yes"]},
        )
        gower1 = Gower(cfg1).fit(X_train1)
        result1 = gower1.transform(np.array([["No"], ["Yes"]], dtype=object))

        X_train2 = np.array([["No"], ["Yes"]], dtype=object)
        cfg2 = Config(
            feature_types={0: "binary_asymmetric"},
            binary_asymmetric_value_order={0: ["No", "Yes"]},
        )
        gower2 = Gower(cfg2).fit(X_train2)
        result2 = gower2.transform(np.array([["No"], ["Yes"]], dtype=object))

        np.testing.assert_array_equal(result1, result2)
        assert result1[0, 0] == 0.0
        assert result1[1, 0] == 1.0

    def test_explicit_order_with_pandas_dataframe(self) -> None:
        """Test explicit ordering works with pandas DataFrame (using indices).

        Training data: ['No']
        Test data: ['Yes'] -> 'Yes' -> 1.0
        """
        X_train = pd.DataFrame({"col": ["No"]})
        X_test = pd.DataFrame({"col": ["Yes"]})

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            binary_asymmetric_value_order={0: ["No", "Yes"]},
        )
        gower = Gower(cfg).fit(X_train)
        result = gower.transform(X_test)

        assert result.iloc[0, 0] == 1.0

    def test_auto_detect_without_explicit_order(self) -> None:
        """Test that auto-detection still works when no explicit order provided.

        Training data: ['A', 'B']
        Test data: ['A', 'B']

        Should still work with alphabetical ordering: 'A'=0.0, 'B'=1.0
        """
        X_train = np.array([["A"], ["B"]], dtype=object)
        X_test = np.array([["A"], ["B"]], dtype=object)

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
        )
        gower = Gower(cfg).fit(X_train)
        result = gower.transform(X_test)

        assert result[0, 0] == 0.0
        assert result[1, 0] == 1.0

    def test_auto_detect_unseen_respects_strategy(self) -> None:
        """Test that auto-detection (no explicit order) respects handle_unseen strategy.

        Training data: ['A', 'A']
        Test data: ['B']

        Strategy 'missing' - should map to nan silently
        """
        X_train = np.array([["A"], ["A"]], dtype=object)
        X_test = np.array([["B"]], dtype=object)
        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            handle_unseen_binary_asymmetric="missing",
        )
        gower = Gower(cfg).fit(X_train)
        result = gower.transform(X_test)

        assert np.isnan(result[0, 0])

    def test_auto_detect_raises_on_too_many_total_values(self) -> None:
        """Test that auto-detection raises error when total unique values > 2.

        Training data: ['A', 'A']
        Test data: ['B', 'C']

        Should raise error about 3 total unique values
        """
        X_train = np.array([["A"], ["A"]], dtype=object)
        X_test = np.array([["B"], ["C"]], dtype=object)

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            handle_unseen_binary_asymmetric="missing",
        )
        gower = Gower(cfg).fit(X_train)

        with pytest.raises(
            ValueError,
            match=r"Binary asymmetric column 0 has 3 unique values total.*fitted: \['A'\], unseen: \['B', 'C'\]",
        ):
            gower.transform(X_test)


class TestBinaryAsymmetricValueOrderValidation:
    """Test config validation for binary_asymmetric_value_order."""

    def test_validation_wrong_number_of_values(self) -> None:
        """Test that providing != 2 values raises validation error."""
        with pytest.raises(ValidationError, match=r"must have exactly 2 values"):
            Config(
                feature_types={0: "binary_asymmetric"},
                binary_asymmetric_value_order={0: ["Only", "One", "Three"]},
            )

    def test_validation_single_value(self) -> None:
        """Test that providing 1 value raises validation error."""
        with pytest.raises(ValidationError, match=r"must have exactly 2 values"):
            Config(
                feature_types={0: "binary_asymmetric"},
                binary_asymmetric_value_order={0: ["OnlyOne"]},
            )

    def test_validation_duplicate_values(self) -> None:
        """Test that duplicate values raise validation error."""
        with pytest.raises(ValidationError, match=r"must be unique"):
            Config(
                feature_types={0: "binary_asymmetric"},
                binary_asymmetric_value_order={0: ["Same", "Same"]},
            )

    def test_validation_non_binary_column(self) -> None:
        """Test that specifying order for non-binary column raises error."""
        with pytest.raises(
            ValidationError,
            match=r"contains non-binary_asymmetric columns",
        ):
            Config(
                feature_types={0: "categorical_nominal"},
                binary_asymmetric_value_order={0: ["A", "B"]},
            )

    def test_validation_not_a_list(self) -> None:
        """Test that non-list values raise validation error."""
        with pytest.raises(ValidationError, match=r"Input should be a valid list"):
            Config(
                feature_types={0: "binary_asymmetric"},
                binary_asymmetric_value_order={0: "NotAList"},
            )

    def test_validation_none_is_valid(self) -> None:
        """Test that None is a valid value (uses auto-detection)."""
        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            binary_asymmetric_value_order=None,
        )
        assert cfg.binary_asymmetric_value_order is None

    def test_validation_multiple_columns(self) -> None:
        """Test validation with multiple binary columns."""
        cfg = Config(
            feature_types={
                0: "binary_asymmetric",
                1: "binary_asymmetric",
                2: "categorical_nominal",
            },
            binary_asymmetric_value_order={
                0: ["No", "Yes"],
                1: [False, True],
            },
        )
        assert cfg.binary_asymmetric_value_order[0] == ["No", "Yes"]
        assert cfg.binary_asymmetric_value_order[1] == [False, True]
