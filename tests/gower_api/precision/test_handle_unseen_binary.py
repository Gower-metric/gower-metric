"""Tests for handle_unseen_binary_asymmetric and handle_unseen_binary_symmetric parameters."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gower_metric import Config, Gower

BINARY_TYPES = ["binary_asymmetric", "binary_symmetric"]


def _config(binary_type: str, strategy: str | None = ..., **extra: Any) -> Config:  # type: ignore[assignment]
    """Build a Config for a single binary column with the given unseen strategy."""
    kw: dict[str, Any] = {"feature_types": extra.pop("feature_types", {0: binary_type})}
    if strategy is not ...:
        kw[f"handle_unseen_{binary_type}"] = strategy
    kw.update(extra)
    return Config(**kw)


class TestHandleUnseenBinary:
    """Test unseen value handling for binary features (asymmetric + symmetric)."""

    @pytest.fixture(autouse=True, params=BINARY_TYPES)
    def setup_binary_type(self, request: pytest.FixtureRequest) -> None:
        self.binary_type: str = request.param

    def test_strategy_error_raises_on_unseen_value(self) -> None:
        """handle_unseen='error' raises ValueError on unseen value."""
        X_train = np.array([["A"], ["A"], ["A"]], dtype=object)
        X_test = np.array([["A"], ["B"]], dtype=object)

        gower = Gower(_config(self.binary_type, "error")).fit(X_train)

        with pytest.raises(
            ValueError,
            match=r"Value 'B' in column 0 not found in fitted binary mapping",
        ):
            gower.transform(X_test)

    def test_strategy_warning_maps_to_nan(self) -> None:
        """handle_unseen='warning' maps unseen to NaN with warning."""
        X_train = np.array([["A"], ["A"], ["A"]], dtype=object)
        X_test = np.array([["A"], ["B"]], dtype=object)

        gower = Gower(_config(self.binary_type, "warning")).fit(X_train)

        with pytest.warns(
            UserWarning,
            match=r"Value 'B' in column 0 not found.*Treating as missing",
        ):
            result = gower.transform(X_test)

        assert result[0, 0] == 0.0
        assert np.isnan(result[1, 0])

    def test_strategy_missing_maps_to_nan_silently(self) -> None:
        """handle_unseen='missing' maps unseen to NaN without warning."""
        X_train = np.array([["A"], ["A"]], dtype=object)
        X_test = np.array([["A"], ["B"]], dtype=object)

        gower = Gower(_config(self.binary_type, "missing")).fit(X_train)
        result = gower.transform(X_test)

        assert result[0, 0] == 0.0
        assert np.isnan(result[1, 0])

    def test_default_strategy_is_error(self) -> None:
        """Default strategy is 'error' — unseen value raises ValueError."""
        X_train = np.array([["Yes"], ["Yes"]], dtype=object)
        X_test = np.array([["No"]], dtype=object)

        gower = Gower(_config(self.binary_type)).fit(X_train)

        with pytest.raises(
            ValueError,
            match=r"Value 'No' in column 0 not found in fitted binary mapping",
        ):
            gower.transform(X_test)

    def test_complete_fit_with_strategy_error(self) -> None:
        """Third value after complete fit violates binary constraint."""
        X_train = np.array([["A"], ["B"], ["A"], ["B"]], dtype=object)
        X_test = np.array([["C"]], dtype=object)

        gower = Gower(_config(self.binary_type, "error")).fit(X_train)

        with pytest.raises(ValueError, match=r"has 3 unique values total"):
            gower.transform(X_test)

    def test_complete_fit_with_strategy_missing(self) -> None:
        """Third value with 'missing' strategy still raises (3 total values)."""
        X_train = np.array([[0], [1], [0], [1]], dtype=object)
        X_test = np.array([[2]], dtype=object)

        gower = Gower(_config(self.binary_type, "missing")).fit(X_train)

        with pytest.raises(ValueError, match=r"has 3 unique values total"):
            gower.transform(X_test)

    def test_invalid_strategy_raises_validation_error(self) -> None:
        """Invalid strategy string raises pydantic ValidationError."""
        with pytest.raises(
            ValidationError,
            match=r"Input should be 'warning', 'error' or 'missing'",
        ):
            _config(self.binary_type, "invalid")  # type: ignore[arg-type]

    def test_strategy_with_pandas_dataframe(self) -> None:
        """Strategy works with pandas DataFrame input."""
        X_train = pd.DataFrame({"col": ["A", "A"]})
        X_test = pd.DataFrame({"col": ["B"]})

        cfg = _config(
            self.binary_type,
            "missing",
            feature_types={"col": self.binary_type},
        )
        gower = Gower(cfg).fit(X_train)

        result = gower.transform(X_test)
        assert np.isnan(result.iloc[0, 0])  # type: ignore[union-attr]

    def test_strategy_with_multiple_unseen_values(self) -> None:
        """Multiple unseen values violate binary constraint (4 total)."""
        X_train = np.array([["A"], ["A"], ["A"]], dtype=object)
        X_test = np.array([["B"], ["C"], ["D"]], dtype=object)

        gower = Gower(_config(self.binary_type, "missing")).fit(X_train)

        with pytest.raises(ValueError, match=r"has 4 unique values total"):
            gower.transform(X_test)
