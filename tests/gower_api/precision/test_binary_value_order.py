"""Tests for binary_*_value_order parameters (asymmetric + symmetric, parametrized)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gower_metric import Config, Gower
from tests.gower_api.precision.conftest import BaseTest

BINARY_TYPES = ["binary_asymmetric", "binary_symmetric"]


class TestBinaryValueOrder(BaseTest):
    """Value-order tests parametrized by binary type and float dtype."""

    @pytest.fixture(autouse=True, params=BINARY_TYPES)
    def setup_binary_type(self, request) -> None:
        self.binary_type: str = request.param

    def _config(
        self,
        feature_types: dict[int | str, str] | None = None,
        value_order: dict[int, list[Any]] | None = ...,  # type: ignore[assignment]
        handle_unseen: str | None = None,
    ) -> Config:
        kw: dict[str, Any] = {"data_type": self.dtype}
        kw["feature_types"] = feature_types or {0: self.binary_type}
        if value_order is not ...:
            kw[f"{self.binary_type}_value_order"] = value_order
        if handle_unseen is not None:
            kw[f"handle_unseen_{self.binary_type}"] = handle_unseen
        return Config(**kw)

    def test_explicit_order_complete_fit(self) -> None:
        """Explicit ordering when training has both values."""
        X_train = np.array([["No"], ["Yes"], ["No"]], dtype=object)
        X_test = np.array([["Yes"], ["No"]], dtype=object)

        gower = Gower(self._config(value_order={0: ["No", "Yes"]})).fit(X_train)
        result = gower.transform(X_test)

        assert result[0, 0] == 1.0
        assert result[1, 0] == 0.0

    def test_explicit_order_degenerate_fit_sees_second_value(self) -> None:
        """Explicit ordering handles expected-but-not-yet-seen values."""
        X_train = np.array([["No"], ["No"]], dtype=object)
        X_test = np.array([["Yes"]], dtype=object)

        gower = Gower(self._config(value_order={0: ["No", "Yes"]})).fit(X_train)
        result = gower.transform(X_test)

        assert result[0, 0] == 1.0

    def test_explicit_order_truly_unseen_value_always_raises(self) -> None:
        """Unseen values violating explicit order ALWAYS raise ValueError."""
        X_train = np.array([["No"], ["Yes"]], dtype=object)
        X_test = np.array([["Maybe"]], dtype=object)

        gower = Gower(
            self._config(value_order={0: ["No", "Yes"]}, handle_unseen="missing"),
        ).fit(X_train)

        with pytest.raises(
            ValueError,
            match=rf"Value 'Maybe' in column 0 violates {self.binary_type}_value_order",
        ):
            gower.transform(X_test)

    def test_explicit_order_violation_ignores_strategy(self) -> None:
        """Explicit order violations raise error regardless of strategy."""
        X_train = np.array([["A"], ["B"]], dtype=object)
        X_test = np.array([["C"]], dtype=object)

        for strategy in ["error", "warning", "missing"]:
            gower = Gower(
                self._config(value_order={0: ["A", "B"]}, handle_unseen=strategy),
            ).fit(X_train)

            with pytest.raises(
                ValueError,
                match=rf"violates {self.binary_type}_value_order",
            ):
                gower.transform(X_test)

    def test_explicit_order_training_has_unexpected_value(self) -> None:
        """Training data with unexpected values raises error."""
        X_train = np.array([["No"], ["Maybe"]], dtype=object)

        with pytest.raises(
            ValueError,
            match=r"contains unexpected values.*Maybe.*Expected values",
        ):
            Gower(self._config(value_order={0: ["No", "Yes"]})).fit(X_train)

    def test_explicit_order_reversed_maintains_consistency(self) -> None:
        """Explicit ordering is not affected by data order."""
        cfg = self._config(value_order={0: ["No", "Yes"]})

        gower1 = Gower(cfg).fit(np.array([["Yes"], ["No"]], dtype=object))
        result1 = gower1.transform(np.array([["No"], ["Yes"]], dtype=object))

        gower2 = Gower(cfg).fit(np.array([["No"], ["Yes"]], dtype=object))
        result2 = gower2.transform(np.array([["No"], ["Yes"]], dtype=object))

        np.testing.assert_array_equal(result1, result2)
        assert result1[0, 0] == 0.0
        assert result1[1, 0] == 1.0

    def test_explicit_order_with_pandas_dataframe(self) -> None:
        """Explicit ordering works with pandas DataFrame."""
        X_train = pd.DataFrame({"col": ["No"]})
        X_test = pd.DataFrame({"col": ["Yes"]})

        gower = Gower(self._config(value_order={0: ["No", "Yes"]})).fit(X_train)
        result = gower.transform(X_test)

        assert result.iloc[0, 0] == 1.0  # type: ignore[union-attr]

    def test_auto_detect_without_explicit_order(self) -> None:
        """Auto-detection works when no explicit order provided."""
        X_train = np.array([["A"], ["B"]], dtype=object)
        X_test = np.array([["A"], ["B"]], dtype=object)

        gower = Gower(self._config()).fit(X_train)
        result = gower.transform(X_test)

        assert result[0, 0] == 0.0
        assert result[1, 0] == 1.0

    def test_auto_detect_unseen_respects_strategy(self) -> None:
        """Auto-detection respects handle_unseen strategy."""
        X_train = np.array([["A"], ["A"]], dtype=object)
        X_test = np.array([["B"]], dtype=object)

        gower = Gower(self._config(handle_unseen="missing")).fit(X_train)
        result = gower.transform(X_test)

        assert np.isnan(result[0, 0])

    def test_auto_detect_raises_on_too_many_total_values(self) -> None:
        """Auto-detection raises error when total unique values > 2."""
        short_type = self.binary_type.replace("binary_", "")
        X_train = np.array([["A"], ["A"]], dtype=object)
        X_test = np.array([["B"], ["C"]], dtype=object)

        gower = Gower(self._config(handle_unseen="missing")).fit(X_train)

        with pytest.raises(
            ValueError,
            match=rf"Binary {short_type} column 0 has 3 unique values total.*fitted: \['A'\], unseen: \['B', 'C'\]",
        ):
            gower.transform(X_test)


class TestBinaryValueOrderValidation(BaseTest):
    """Config validation tests parametrized by binary type and float dtype."""

    @pytest.fixture(autouse=True, params=BINARY_TYPES)
    def setup_binary_type(self, request) -> None:
        self.binary_type: str = request.param

    def _config(self, **kw: Any) -> Config:
        cfg_kw: dict[str, Any] = {"data_type": self.dtype}
        cfg_kw.update(kw)
        return Config(**cfg_kw)

    def test_validation_wrong_number_of_values(self) -> None:
        with pytest.raises(ValidationError, match=r"must have exactly 2 values"):
            self._config(
                feature_types={0: self.binary_type},
                **{f"{self.binary_type}_value_order": {0: ["Only", "One", "Three"]}},
            )

    def test_validation_single_value(self) -> None:
        with pytest.raises(ValidationError, match=r"must have exactly 2 values"):
            self._config(
                feature_types={0: self.binary_type},
                **{f"{self.binary_type}_value_order": {0: ["OnlyOne"]}},
            )

    def test_validation_duplicate_values(self) -> None:
        with pytest.raises(ValidationError, match=r"must be unique"):
            self._config(
                feature_types={0: self.binary_type},
                **{f"{self.binary_type}_value_order": {0: ["Same", "Same"]}},
            )

    def test_validation_non_binary_column(self) -> None:
        with pytest.raises(
            ValidationError,
            match=rf"contains non-{self.binary_type} columns",
        ):
            self._config(
                feature_types={0: "categorical_nominal"},
                **{f"{self.binary_type}_value_order": {0: ["A", "B"]}},
            )

    def test_validation_not_a_list(self) -> None:
        with pytest.raises(ValidationError, match=r"Input should be a valid list"):
            self._config(
                feature_types={0: self.binary_type},
                **{f"{self.binary_type}_value_order": {0: "NotAList"}},
            )

    def test_validation_none_is_valid(self) -> None:
        cfg = self._config(
            feature_types={0: self.binary_type},
            **{f"{self.binary_type}_value_order": None},
        )
        assert getattr(cfg, f"{self.binary_type}_value_order") is None

    def test_validation_multiple_columns(self) -> None:
        cfg = self._config(
            feature_types={
                0: self.binary_type,
                1: self.binary_type,
                2: "categorical_nominal",
            },
            **{
                f"{self.binary_type}_value_order": {
                    0: ["No", "Yes"],
                    1: [False, True],
                },
            },
        )
        vo = getattr(cfg, f"{self.binary_type}_value_order")
        assert vo[0] == ["No", "Yes"]
        assert vo[1] == [False, True]
