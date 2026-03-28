"""Tests for handle_unseen_categorical_nominal and handle_unseen_categorical_ordinal parameters."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gower_metric import Config, Gower

CATEGORICAL_TYPES = ["categorical_nominal", "categorical_ordinal"]

# Per-type test data and config extras
_TYPE_DATA: dict[str, dict[str, Any]] = {
    "categorical_nominal": {
        "train_values": ["A", "B", "C"],
        "unseen_value": "D",
        "warning_match": r"Unseen values.*in nominal column 0.*Treating as missing",
    },
    "categorical_ordinal": {
        "train_values": ["low", "medium", "high"],
        "unseen_value": "extra",
        "warning_match": r"Unseen values.*in ordinal column 0.*Treating as missing",
    },
}


def _config(cat_type: str, strategy: str | None = ..., **extra: Any) -> Config:  # type: ignore[assignment]
    """Build a Config for a single categorical column with the given unseen strategy."""
    kw: dict[str, Any] = {"feature_types": extra.pop("feature_types", {0: cat_type})}
    if cat_type == "categorical_ordinal":
        kw["categorical_ordinal_values_order"] = extra.pop(
            "values_order",
            {0: ["low", "medium", "high"]},
        )
    if strategy is not ...:
        kw[f"handle_unseen_{cat_type}"] = strategy
    kw.update(extra)
    return Config(**kw)


class TestHandleUnseenCategorical:
    """Test unseen value handling for categorical features (nominal + ordinal)."""

    @pytest.fixture(autouse=True, params=CATEGORICAL_TYPES)
    def setup_cat_type(self, request: pytest.FixtureRequest) -> None:
        self.cat_type: str = request.param
        data = _TYPE_DATA[self.cat_type]
        self.train_values: list[str] = data["train_values"]
        self.unseen_value: str = data["unseen_value"]
        self.warning_match: str = data["warning_match"]

    def _train(self, values: list[str] | None = None) -> np.ndarray:
        vals = values or self.train_values[:2]
        return np.array([[v] for v in vals], dtype=object)

    def test_default_strategy_is_error(self) -> None:
        """Default strategy is 'error' — unseen value raises ValueError."""
        gower = Gower(_config(self.cat_type)).fit(self._train())

        with pytest.raises(ValueError, match=r"Found unknown categories"):
            gower.transform(np.array([[self.unseen_value]], dtype=object))

    def test_strategy_error_raises_on_unseen_value(self) -> None:
        """Explicit 'error' strategy raises ValueError on unseen value."""
        gower = Gower(_config(self.cat_type, "error")).fit(self._train())

        with pytest.raises(ValueError, match=r"Found unknown categories"):
            gower.transform(np.array([[self.unseen_value]], dtype=object))

    def test_strategy_warning_maps_to_nan(self) -> None:
        """'warning' strategy maps unseen to NaN with UserWarning."""
        seen = self.train_values[0]
        X_test = np.array([[seen], [self.unseen_value]], dtype=object)

        gower = Gower(_config(self.cat_type, "warning")).fit(self._train())

        with pytest.warns(UserWarning, match=self.warning_match):
            result = gower.transform(X_test)

        assert result[0, 0] == 0.0
        assert np.isnan(result[1, 0])

    def test_strategy_missing_maps_to_nan_silently(self) -> None:
        """'missing' strategy maps unseen to NaN without warning."""
        seen = self.train_values[0]
        X_test = np.array([[seen], [self.unseen_value]], dtype=object)

        gower = Gower(_config(self.cat_type, "missing")).fit(self._train())
        result = gower.transform(X_test)

        assert result[0, 0] == 0.0
        assert np.isnan(result[1, 0])

    def test_invalid_strategy_raises_validation_error(self) -> None:
        """Invalid strategy string raises pydantic ValidationError."""
        with pytest.raises(
            ValidationError,
            match=r"Input should be 'warning', 'error' or 'missing'",
        ):
            _config(self.cat_type, "invalid")  # type: ignore[arg-type]

    def test_strategy_with_pandas_dataframe(self) -> None:
        """Strategy works with pandas DataFrame input."""
        col_name = "col"
        train_vals = self.train_values[:2]
        X_train = pd.DataFrame({col_name: train_vals})
        X_test = pd.DataFrame({col_name: [self.unseen_value]})

        kw: dict[str, Any] = {"feature_types": {col_name: self.cat_type}}
        if self.cat_type == "categorical_ordinal":
            kw["values_order"] = {col_name: self.train_values}
        cfg = _config(self.cat_type, "missing", **kw)
        gower = Gower(cfg).fit(X_train)

        result = gower.transform(X_test)
        assert np.isnan(result.iloc[0, 0])  # type: ignore[union-attr]

    def test_all_seen_values_work_fine(self) -> None:
        """All seen values produce no NaN with default error strategy."""
        X = self._train(self.train_values)

        gower = Gower(_config(self.cat_type)).fit(X)
        result = gower.transform(X)

        assert not np.isnan(result).any()


class TestHandleUnseenCategoricalOrdinalSpecific:
    """Ordinal-specific unseen value tests."""

    def test_expected_but_unseen_ordinal_value_works(self) -> None:
        """Value in ordinal order but not in training data works fine."""
        X_train = np.array([["low"], ["medium"]], dtype=object)
        X_test = np.array([["high"]], dtype=object)

        cfg = _config("categorical_ordinal")
        gower = Gower(cfg).fit(X_train)

        result = gower.transform(X_test)
        assert result[0, 0] == 2.0
