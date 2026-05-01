"""Tests for Config field validators - covers every validation error branch."""

import numpy as np
import pytest
from pydantic import ValidationError

from gower_metric import Config
from gower_metric.core.config import OutOfRangeStrategy, SkipOutOfRangeValidation

DEFAULT_DTYPE = np.float64


class TestFeatureTypeValidation:
    def test_invalid_feature_type_raises(self) -> None:
        with pytest.raises(ValidationError, match="Invalid feature type"):
            Config(feature_types={0: "unknown_type"}, data_type=DEFAULT_DTYPE)

    def test_valid_feature_types_pass(self) -> None:
        cfg = Config(feature_types={0: "numeric"}, data_type=DEFAULT_DTYPE)
        assert cfg.feature_types[0] == "numeric"


class TestScaleWindowTypeValidation:
    def test_scale_window_type_without_scale_window_raises(self) -> None:
        """scale_window=None but scale_window_type='silverman' raises ValueError."""
        with pytest.raises(ValidationError, match="scale_window_type must be None"):
            Config(
                feature_types={0: "numeric"},
                scale_window=None,
                scale_window_type="silverman",
                data_type=DEFAULT_DTYPE,
            )

    def test_valid_scale_window_type_with_kde(self) -> None:
        cfg = Config(
            feature_types={0: "numeric"},
            scale_window="kde",
            scale_window_type="silverman",
            data_type=DEFAULT_DTYPE,
        )
        assert cfg.scale_window_type == "silverman"

    def test_kde_with_none_scale_window_type_passes(self) -> None:
        cfg = Config(
            feature_types={0: "numeric"},
            scale_window="kde",
            scale_window_type=None,
            data_type=DEFAULT_DTYPE,
        )
        assert cfg.scale_window_type is None


class TestKNeighborsValidation:
    def test_zero_raises(self) -> None:
        with pytest.raises(
            ValidationError,
            match="k_neighbors must be None or a positive integer",
        ):
            Config(feature_types={0: "numeric"}, k_neighbors=0, data_type=DEFAULT_DTYPE)

    def test_negative_raises(self) -> None:
        with pytest.raises(
            ValidationError,
            match="k_neighbors must be None or a positive integer",
        ):
            Config(
                feature_types={0: "numeric"},
                k_neighbors=-5,
                data_type=DEFAULT_DTYPE,
            )


class TestOrdinalOrderValidation:
    def test_missing_order_for_ordinal_column_raises(self) -> None:
        with pytest.raises(ValidationError, match="must have a values order defined"):
            Config(
                feature_types={0: "categorical_ordinal"},
                categorical_ordinal_values_order=None,
                data_type=DEFAULT_DTYPE,
            )

    def test_partial_order_missing_column_raises(self) -> None:
        with pytest.raises(ValidationError, match="Missing order definitions"):
            Config(
                feature_types={0: "categorical_ordinal", 1: "categorical_ordinal"},
                categorical_ordinal_values_order={0: ["a", "b"]},
                data_type=DEFAULT_DTYPE,
            )


class TestHandleUnseenValidation:
    """Pydantic Literal type checks catch invalid values before our validators run.

    We test that the right error is raised regardless of whether it's Pydantic or our code.
    """

    def test_invalid_handle_unseen_binary_asymmetric(self) -> None:
        with pytest.raises(ValidationError):
            Config(
                feature_types={0: "binary_asymmetric"},
                handle_unseen_binary_asymmetric="bad",  # type: ignore[arg-type]
                data_type=DEFAULT_DTYPE,
            )

    def test_invalid_handle_unseen_binary_symmetric(self) -> None:
        with pytest.raises(ValidationError):
            Config(
                feature_types={0: "binary_symmetric"},
                handle_unseen_binary_symmetric="bad",  # type: ignore[arg-type]
                data_type=DEFAULT_DTYPE,
            )

    def test_invalid_handle_unseen_categorical_nominal(self) -> None:
        with pytest.raises(ValidationError):
            Config(
                feature_types={0: "categorical_nominal"},
                handle_unseen_categorical_nominal="bad",  # type: ignore[arg-type]
                data_type=DEFAULT_DTYPE,
            )

    def test_invalid_handle_unseen_categorical_ordinal(self) -> None:
        with pytest.raises(ValidationError):
            Config(
                feature_types={0: "categorical_ordinal"},
                categorical_ordinal_values_order={0: ["a", "b"]},
                handle_unseen_categorical_ordinal="bad",  # type: ignore[arg-type]
                data_type=DEFAULT_DTYPE,
            )


class TestBinaryValueOrderValidation:
    def test_asymmetric_wrong_count_raises(self) -> None:
        with pytest.raises(ValidationError, match="exactly 2 values"):
            Config(
                feature_types={0: "binary_asymmetric"},
                binary_asymmetric_value_order={0: [0, 1, 2]},
                data_type=DEFAULT_DTYPE,
            )

    def test_asymmetric_duplicate_values_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be unique"):
            Config(
                feature_types={0: "binary_asymmetric"},
                binary_asymmetric_value_order={0: [1, 1]},
                data_type=DEFAULT_DTYPE,
            )

    def test_asymmetric_extra_column_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-binary_asymmetric"):
            Config(
                feature_types={0: "binary_asymmetric"},
                binary_asymmetric_value_order={0: [0, 1], 1: [0, 1]},
                data_type=DEFAULT_DTYPE,
            )

    def test_symmetric_wrong_count_raises(self) -> None:
        with pytest.raises(ValidationError, match="exactly 2 values"):
            Config(
                feature_types={0: "binary_symmetric"},
                binary_symmetric_value_order={0: [0, 1, 2]},
                data_type=DEFAULT_DTYPE,
            )

    def test_symmetric_duplicate_values_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be unique"):
            Config(
                feature_types={0: "binary_symmetric"},
                binary_symmetric_value_order={0: [1, 1]},
                data_type=DEFAULT_DTYPE,
            )

    def test_symmetric_extra_column_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-binary_symmetric"):
            Config(
                feature_types={0: "binary_symmetric"},
                binary_symmetric_value_order={0: [0, 1], 1: [0, 1]},
                data_type=DEFAULT_DTYPE,
            )


class TestConditionalDistancesValidation:
    def test_invalid_conditional_distances_type(self) -> None:
        """Pydantic Literal[True, False] catches non-bool values."""
        with pytest.raises(ValidationError):
            Config(
                feature_types={0: "numeric", 1: "categorical_nominal"},
                conditional_distances=2,  # type: ignore[arg-type]
                data_type=DEFAULT_DTYPE,
            )


class TestOutOfRangeConfigValidation:
    """Config validation for out_of_range parameter."""

    @pytest.mark.parametrize("strategy", ["clip", "warning", "error"])
    def test_valid_values(self, strategy: OutOfRangeStrategy) -> None:
        cfg = Config(feature_types={0: "numeric"}, out_of_range=strategy)
        assert cfg.out_of_range == strategy

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"Input should be"):
            Config(feature_types={0: "numeric"}, out_of_range="invalid")  # type: ignore[arg-type]

    def test_default_is_error(self) -> None:
        cfg = Config(feature_types={0: "numeric"})
        assert cfg.out_of_range == "error"


class TestSkipOutOfRangeConfigValidation:
    """Config validation for skip_out_of_range_validation parameter."""

    @pytest.mark.parametrize("flag", [True, False])
    def test_valid_values(self, flag: SkipOutOfRangeValidation) -> None:
        cfg = Config(feature_types={0: "numeric"}, skip_out_of_range_validation=flag)
        assert cfg.skip_out_of_range_validation == flag

    @pytest.mark.parametrize("invalid_value", ["no", "yes", 1, 0])
    def test_invalid_value_raises(self, invalid_value: object) -> None:
        with pytest.raises(ValidationError, match=r"Input should be a valid boolean"):
            Config(
                feature_types={0: "numeric"},
                skip_out_of_range_validation=invalid_value,  # type: ignore[arg-type]
            )

    def test_default_is_false(self) -> None:
        cfg = Config(feature_types={0: "numeric"})
        assert cfg.skip_out_of_range_validation is False


class TestSilvermanConstantConfigValidation:
    """Config validation for silverman_constant parameter."""

    def test_default_no_scale_window_passes(self) -> None:
        cfg = Config(feature_types={0: "numeric"})
        assert cfg.silverman_constant == 1.06

    def test_default_with_knn_passes(self) -> None:
        cfg = Config(feature_types={0: "numeric"}, scale_window="kNN")
        assert cfg.silverman_constant == 1.06

    @pytest.mark.parametrize("c", [0.9, 1.06, 0.5, 2.0, 1])
    def test_valid_values_with_full_silverman_kde(self, c: float) -> None:
        cfg = Config(
            feature_types={0: "numeric"},
            silverman_constant=c,
            scale_window="kde",
            scale_window_type="silverman",
        )
        assert cfg.silverman_constant == c

    def test_set_constant_without_kde_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"requires scale_window='kde'"):
            Config(feature_types={0: "numeric"}, silverman_constant=0.9)

    def test_set_constant_with_knn_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"requires scale_window='kde'"):
            Config(
                feature_types={0: "numeric"},
                silverman_constant=0.9,
                scale_window="kNN",
            )

    def test_set_constant_kde_without_silverman_type_raises(self) -> None:
        with pytest.raises(
            ValidationError,
            match=r"requires scale_window_type='silverman'",
        ):
            Config(
                feature_types={0: "numeric"},
                silverman_constant=0.9,
                scale_window="kde",
            )

    @pytest.mark.parametrize("invalid_value", [0, -1.0, -0.001])
    def test_non_positive_value_raises(self, invalid_value: float) -> None:
        with pytest.raises(ValidationError, match=r"must be a positive number"):
            Config(
                feature_types={0: "numeric"},
                silverman_constant=invalid_value,
                scale_window="kde",
                scale_window_type="silverman",
            )
