"""Tests for handle_unseen_binary_asymmetric parameter."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gower_metric import Config, Gower


class TestHandleUnseenBinaryAsymmetric:
    """Test unseen value handling for binary_asymmetric features."""

    def test_strategy_error_raises_on_unseen_value(self) -> None:
        """Test that handle_unseen_binary_asymmetric='error' raises ValueError.

        Training data: only 'A'
        Test data: has 'B' (unseen) -> ValueError
        """
        X_train = np.array([["A"], ["A"], ["A"]], dtype=object)
        X_test = np.array([["A"], ["B"]], dtype=object)

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            handle_unseen_binary_asymmetric="error",
        )
        gower = Gower(cfg).fit(X_train)

        with pytest.raises(
            ValueError,
            match=r"Value 'B' in column 0 not found in fitted binary mapping",
        ):
            gower.transform(X_test)

    def test_strategy_warning_maps_to_nan(self) -> None:
        """Test that handle_unseen_binary_asymmetric='warning' maps to nan with warning.

        Training data: only 'A'
        Test data: has 'B' (unseen) -> warning, nan
        """
        X_train = np.array([["A"], ["A"], ["A"]], dtype=object)
        X_test = np.array([["A"], ["B"]], dtype=object)

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            handle_unseen_binary_asymmetric="warning",
        )
        gower = Gower(cfg).fit(X_train)

        with pytest.warns(
            UserWarning,
            match=r"Value 'B' in column 0 not found.*Treating as missing",
        ):
            result = gower.transform(X_test)

        assert result[0, 0] == 0.0
        assert np.isnan(result[1, 0])

    def test_strategy_missing_maps_to_nan_silently(self) -> None:
        """Test that handle_unseen_binary_asymmetric='missing' maps to nan without warning.

        Training data: only 'A'
        Test data: has 'B' (unseen) -> nan
        """
        X_train = np.array([["A"], ["A"]], dtype=object)
        X_test = np.array([["A"], ["B"]], dtype=object)

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            handle_unseen_binary_asymmetric="missing",
        )
        gower = Gower(cfg).fit(X_train)

        result = gower.transform(X_test)

        assert result[0, 0] == 0.0
        assert np.isnan(result[1, 0])

    def test_default_strategy_is_error(self) -> None:
        """Test that default strategy is 'error'.

        Training data: only 'Yes'
        Test data: has 'No' (unseen) -> ValueError
        """
        X_train = np.array([["Yes"], ["Yes"]], dtype=object)
        X_test = np.array([["No"]], dtype=object)

        cfg = Config(feature_types={0: "binary_asymmetric"})
        gower = Gower(cfg).fit(X_train)

        with pytest.raises(
            ValueError,
            match=r"Value 'No' in column 0 not found in fitted binary mapping",
        ):
            gower.transform(X_test)

    def test_complete_fit_with_strategy_error(self) -> None:
        """Test that with complete fit, third value violates binary dtype.

        Training data: 'A' and 'B'
        Test data: 'C' (unseen) -> ValueError because of 3 total values
        """
        X_train = np.array([["A"], ["B"], ["A"], ["B"]], dtype=object)
        X_test = np.array([["C"]], dtype=object)

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            handle_unseen_binary_asymmetric="error",
        )
        gower = Gower(cfg).fit(X_train)

        with pytest.raises(ValueError, match=r"has 3 unique values total"):
            gower.transform(X_test)

    def test_complete_fit_with_strategy_missing(self) -> None:
        """Test complete fit with third value and 'missing' strategy.

        Training data: '0' and '1'
        Test data: '2' (unseen) -> ValueError because of 3 total values
        """
        X_train = np.array([[0], [1], [0], [1]], dtype=object)
        X_test = np.array([[2]], dtype=object)

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            handle_unseen_binary_asymmetric="missing",
        )
        gower = Gower(cfg).fit(X_train)

        with pytest.raises(ValueError, match=r"has 3 unique values total"):
            gower.transform(X_test)

    def test_invalid_strategy_raises_validation_error(self) -> None:
        """Test that invalid strategy raises validation error."""
        with pytest.raises(
            ValidationError,
            match=r"Input should be 'warning', 'error' or 'missing'",
        ):
            Config(
                feature_types={0: "binary_asymmetric"},
                handle_unseen_binary_asymmetric="invalid",  # type: ignore[arg-type]
            )

    def test_strategy_with_pandas_dataframe(self) -> None:
        """Test that strategy works with pandas DataFrame."""
        X_train = pd.DataFrame({"col": ["A", "A"]})
        X_test = pd.DataFrame({"col": ["B"]})

        cfg = Config(
            feature_types={"col": "binary_asymmetric"},
            handle_unseen_binary_asymmetric="missing",
        )
        gower = Gower(cfg).fit(X_train)

        result = gower.transform(X_test)
        assert np.isnan(result.iloc[0, 0])  # type: ignore[union-attr]

    def test_strategy_with_multiple_unseen_values(self) -> None:
        """Test that multiple unseen values violate binary dtype.

        Training data: only 'A'
        Test data: 'B', 'C', 'D' (unseen) -> ValueError because of 4 total values
        """
        X_train = np.array([["A"], ["A"], ["A"]], dtype=object)
        X_test = np.array([["B"], ["C"], ["D"]], dtype=object)

        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            handle_unseen_binary_asymmetric="missing",
        )
        gower = Gower(cfg).fit(X_train)

        with pytest.raises(ValueError, match=r"has 4 unique values total"):
            gower.transform(X_test)
