"""Tests for handle_unseen_categorical_nominal parameter."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gower_metric import Config, Gower


class TestHandleUnseenCategoricalNominal:
    """Test unseen value handling for categorical_nominal features."""

    def test_default_strategy_is_error(self) -> None:
        """Test that default strategy is 'error'.

        Training data: ['A', 'B']
        Test data: has 'C' (unseen) -> ValueError
        """
        X_train = np.array([["A"], ["B"]], dtype=object)
        X_test = np.array([["C"]], dtype=object)

        cfg = Config(feature_types={0: "categorical_nominal"})
        gower = Gower(cfg).fit(X_train)

        with pytest.raises(ValueError, match=r"Found unknown categories"):
            gower.transform(X_test)

    def test_strategy_error_raises_on_unseen_value(self) -> None:
        """Test that handle_unseen_categorical_nominal='error' raises ValueError."""
        X_train = np.array([["A"], ["B"], ["C"]], dtype=object)
        X_test = np.array([["D"]], dtype=object)

        cfg = Config(
            feature_types={0: "categorical_nominal"},
            handle_unseen_categorical_nominal="error",
        )
        gower = Gower(cfg).fit(X_train)

        with pytest.raises(ValueError, match=r"Found unknown categories"):
            gower.transform(X_test)

    def test_strategy_warning_maps_to_nan(self) -> None:
        """Test that handle_unseen_categorical_nominal='warning' maps to nan with warning."""
        X_train = np.array([["A"], ["B"]], dtype=object)
        X_test = np.array([["A"], ["C"]], dtype=object)

        cfg = Config(
            feature_types={0: "categorical_nominal"},
            handle_unseen_categorical_nominal="warning",
        )
        gower = Gower(cfg).fit(X_train)

        with pytest.warns(
            UserWarning,
            match=r"Unseen values.*in nominal column 0.*Treating as missing",
        ):
            result = gower.transform(X_test)

        assert result[0, 0] == 0.0
        assert np.isnan(result[1, 0])

    def test_strategy_missing_maps_to_nan_silently(self) -> None:
        """Test that handle_unseen_categorical_nominal='missing' maps to nan without warning."""
        X_train = np.array([["A"], ["B"]], dtype=object)
        X_test = np.array([["A"], ["C"]], dtype=object)

        cfg = Config(
            feature_types={0: "categorical_nominal"},
            handle_unseen_categorical_nominal="missing",
        )
        gower = Gower(cfg).fit(X_train)

        result = gower.transform(X_test)

        assert result[0, 0] == 0.0
        assert np.isnan(result[1, 0])

    def test_invalid_strategy_raises_validation_error(self) -> None:
        """Test that invalid strategy raises validation error."""
        with pytest.raises(
            ValidationError,
            match=r"Input should be 'warning', 'error' or 'missing'",
        ):
            Config(
                feature_types={0: "categorical_nominal"},
                handle_unseen_categorical_nominal="invalid",
            )

    def test_strategy_with_pandas_dataframe(self) -> None:
        """Test that strategy works with pandas DataFrame."""
        X_train = pd.DataFrame({"col": ["A", "B"]})
        X_test = pd.DataFrame({"col": ["C"]})

        cfg = Config(
            feature_types={"col": "categorical_nominal"},
            handle_unseen_categorical_nominal="missing",
        )
        gower = Gower(cfg).fit(X_train)

        result = gower.transform(X_test)
        assert np.isnan(result.iloc[0, 0])

    def test_all_seen_values_work_fine(self) -> None:
        """Test that seen values work correctly with error strategy (default)."""
        X_train = np.array([["A"], ["B"], ["C"]], dtype=object)
        X_test = np.array([["A"], ["B"], ["C"]], dtype=object)

        cfg = Config(feature_types={0: "categorical_nominal"})
        gower = Gower(cfg).fit(X_train)

        result = gower.transform(X_test)

        assert not np.isnan(result).any()
