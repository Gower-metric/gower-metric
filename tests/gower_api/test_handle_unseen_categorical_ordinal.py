"""Tests for handle_unseen_categorical_ordinal parameter."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gower_metric import Config, Gower


class TestHandleUnseenCategoricalOrdinal:
    """Test unseen value handling for categorical_ordinal features."""

    def test_default_strategy_is_error(self) -> None:
        """Test that default strategy is 'error'.

        Training data: ['low', 'medium']
        Test data: has 'extra' (unseen, not in order) -> ValueError
        """
        X_train = np.array([["low"], ["medium"]], dtype=object)
        X_test = np.array([["extra"]], dtype=object)

        cfg = Config(
            feature_types={0: "categorical_ordinal"},
            categorical_ordinal_values_order={0: ["low", "medium", "high"]},
        )
        gower = Gower(cfg).fit(X_train)

        with pytest.raises(ValueError, match=r"Found unknown categories"):
            gower.transform(X_test)

    def test_strategy_error_raises_on_unseen_value(self) -> None:
        """Test that handle_unseen_categorical_ordinal='error' raises ValueError."""
        X_train = np.array([["low"], ["medium"]], dtype=object)
        X_test = np.array([["extra"]], dtype=object)

        cfg = Config(
            feature_types={0: "categorical_ordinal"},
            categorical_ordinal_values_order={0: ["low", "medium", "high"]},
            handle_unseen_categorical_ordinal="error",
        )
        gower = Gower(cfg).fit(X_train)

        with pytest.raises(ValueError, match=r"Found unknown categories"):
            gower.transform(X_test)

    def test_strategy_warning_maps_to_nan(self) -> None:
        """Test that handle_unseen_categorical_ordinal='warning' maps to nan with warning."""
        X_train = np.array([["low"], ["medium"]], dtype=object)
        X_test = np.array([["low"], ["extra"]], dtype=object)

        cfg = Config(
            feature_types={0: "categorical_ordinal"},
            categorical_ordinal_values_order={0: ["low", "medium", "high"]},
            handle_unseen_categorical_ordinal="warning",
        )
        gower = Gower(cfg).fit(X_train)

        with pytest.warns(
            UserWarning,
            match=r"Unseen values.*in ordinal column 0.*Treating as missing",
        ):
            result = gower.transform(X_test)

        assert result[0, 0] == 0.0
        assert np.isnan(result[1, 0])

    def test_strategy_missing_maps_to_nan_silently(self) -> None:
        """Test that handle_unseen_categorical_ordinal='missing' maps to nan without warning."""
        X_train = np.array([["low"], ["medium"]], dtype=object)
        X_test = np.array([["low"], ["extra"]], dtype=object)

        cfg = Config(
            feature_types={0: "categorical_ordinal"},
            categorical_ordinal_values_order={0: ["low", "medium", "high"]},
            handle_unseen_categorical_ordinal="missing",
        )
        gower = Gower(cfg).fit(X_train)

        result = gower.transform(X_test)

        assert result[0, 0] == 0.0
        assert np.isnan(result[1, 0])

    def test_expected_but_unseen_ordinal_value_works(self) -> None:
        """Test that values in ordinal order but not in training work fine.

        Training data: ['low', 'medium']
        Test data: ['high'] -> should work because 'high' is in the defined order
        """
        X_train = np.array([["low"], ["medium"]], dtype=object)
        X_test = np.array([["high"]], dtype=object)

        cfg = Config(
            feature_types={0: "categorical_ordinal"},
            categorical_ordinal_values_order={0: ["low", "medium", "high"]},
        )
        gower = Gower(cfg).fit(X_train)

        result = gower.transform(X_test)
        assert result[0, 0] == 2.0

    def test_invalid_strategy_raises_validation_error(self) -> None:
        """Test that invalid strategy raises validation error."""
        with pytest.raises(
            ValidationError,
            match=r"Input should be 'warning', 'error' or 'missing'",
        ):
            Config(
                feature_types={0: "categorical_ordinal"},
                categorical_ordinal_values_order={0: ["low", "medium", "high"]},
                handle_unseen_categorical_ordinal="invalid",
            )

    def test_strategy_with_pandas_dataframe(self) -> None:
        """Test that strategy works with pandas DataFrame."""
        X_train = pd.DataFrame({"ord": ["low", "medium"]})
        X_test = pd.DataFrame({"ord": ["extra"]})

        cfg = Config(
            feature_types={"ord": "categorical_ordinal"},
            categorical_ordinal_values_order={"ord": ["low", "medium", "high"]},
            handle_unseen_categorical_ordinal="missing",
        )
        gower = Gower(cfg).fit(X_train)

        result = gower.transform(X_test)
        assert np.isnan(result.iloc[0, 0])

    def test_all_seen_values_work_fine(self) -> None:
        """Test that seen values work correctly with error strategy (default)."""
        X_train = np.array([["low"], ["medium"], ["high"]], dtype=object)
        X_test = np.array([["low"], ["medium"], ["high"]], dtype=object)

        cfg = Config(
            feature_types={0: "categorical_ordinal"},
            categorical_ordinal_values_order={0: ["low", "medium", "high"]},
        )
        gower = Gower(cfg).fit(X_train)

        result = gower.transform(X_test)
        assert not np.isnan(result).any()
