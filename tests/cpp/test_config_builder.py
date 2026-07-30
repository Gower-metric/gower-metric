"""Tests for build_cpp_config - picks the native config class by data dtype."""

from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import pytest

from gower_metric import Config, Gower
from gower_metric.utils.cpp_middleware.config_builder import _value_orders
from tests.conftest import EDUCATION_LEVELS, generate_mixed_df
from tests.cpp.conftest import (
    DISCRETIZATIONS,
    FEATURE_TYPES,
    MISSING_STRATEGIES,
    MIXED_FEATURE_TYPES,
    SCALE_METHODS,
)

N_MIXED_FEATURES = len(MIXED_FEATURE_TYPES)


def _cpp(gower: Gower) -> Any:
    """Return the built native config, narrowed from None for the type checker."""
    cfg = gower.cpp_config
    assert cfg is not None
    return cfg


def _encoded_rows(gower: Gower, df: pd.DataFrame, dtype) -> np.ndarray:
    """transform(df) as a C-contiguous array the native engine accepts."""
    return np.ascontiguousarray(np.asarray(gower.transform(df)), dtype=dtype)


class TestBuiltFromFittedGower:
    def test_built_after_fit(self, make_gower, expected_config_cls) -> None:
        cfg = make_gower().cpp_config
        assert cfg is not None
        assert isinstance(cfg, expected_config_cls)

    def test_n_features(self, make_gower) -> None:
        gower = make_gower()
        assert gower.cpp_config.n_features == gower.n_feats == N_MIXED_FEATURES

    def test_feature_weights_match_gower(self, make_gower) -> None:
        gower = make_gower()
        np.testing.assert_allclose(
            np.asarray(gower.cpp_config.feature_weights),
            gower.weights,
            rtol=1e-2,
        )


class TestRangesAndBandwidths:
    def test_ranges_only_on_numeric_and_ratio(self, make_gower) -> None:
        gower = make_gower()
        ranges = np.asarray(gower.cpp_config.ranges)
        scaled = sorted({*gower.numeric_indices, *gower.ratio_scale_indices})
        assert np.flatnonzero(ranges).tolist() == scaled

    def test_no_bandwidths_without_discretization(self, make_gower) -> None:
        assert not np.any(np.asarray(make_gower().cpp_config.bandwidths))

    def test_bandwidths_only_on_numeric_and_ratio_with_silverman(
        self,
        make_gower,
    ) -> None:
        gower = make_gower(discretization="silverman")
        bandwidths = np.asarray(gower.cpp_config.bandwidths)
        scaled = sorted({*gower.numeric_indices, *gower.ratio_scale_indices})
        assert np.flatnonzero(bandwidths).tolist() == scaled


class TestCoverageValidation:
    def test_partial_feature_types_raises(self, dtype, rng) -> None:
        df = generate_mixed_df(32, rng)
        feature_types = {k: v for k, v in MIXED_FEATURE_TYPES.items() if k != "Birth"}
        gower = Gower(
            Config(
                feature_types=feature_types,
                data_type=dtype,
                categorical_ordinal_values_order={"Education": EDUCATION_LEVELS},
            ),
        )
        with pytest.raises(ValueError, match="Missmatched number of feature types"):
            gower.fit(df)


class TestNativeValidation:
    """configure_arguments / string_to_* guards."""

    @pytest.mark.parametrize("feature_type", FEATURE_TYPES)
    def test_each_feature_type_accepted(
        self,
        feature_type,
        make_config_data,
        expected_config_cls,
    ) -> None:
        cfg = expected_config_cls()
        cfg.configure_arguments(make_config_data((feature_type,)))
        assert cfg.n_features == 1

    def test_invalid_feature_type_raises(
        self,
        make_config_data,
        expected_config_cls,
    ) -> None:
        data = make_config_data(("numeric", "not_a_feature_type"))
        with pytest.raises(ValueError, match="Invalid feature type"):
            expected_config_cls().configure_arguments(data)

    @pytest.mark.parametrize("strategy", MISSING_STRATEGIES)
    def test_valid_missing_strategies(
        self,
        strategy,
        make_config_data,
        expected_config_cls,
    ) -> None:
        data = make_config_data()
        data.missing_strategy = strategy
        cfg = expected_config_cls()
        cfg.configure_arguments(data)
        assert cfg.n_features == 2

    def test_invalid_missing_strategy_raises(
        self,
        make_config_data,
        expected_config_cls,
    ) -> None:
        data = make_config_data()
        data.missing_strategy = "bogus"
        with pytest.raises(ValueError, match="Invalid missing strategy"):
            expected_config_cls().configure_arguments(data)

    @pytest.mark.parametrize("method", SCALE_METHODS)
    def test_valid_scale_methods(
        self,
        method,
        make_config_data,
        expected_config_cls,
    ) -> None:
        data = make_config_data()
        data.scale_method = method
        cfg = expected_config_cls()
        cfg.configure_arguments(data)
        assert cfg.n_features == 2

    def test_invalid_scale_method_raises(
        self,
        make_config_data,
        expected_config_cls,
    ) -> None:
        data = make_config_data()
        data.scale_method = "bogus"
        with pytest.raises(ValueError, match="Invalid scale method"):
            expected_config_cls().configure_arguments(data)

    def test_weights_length_mismatch_raises(
        self,
        dtype,
        make_config_data,
        expected_config_cls,
    ) -> None:
        data = make_config_data()
        data.feature_weights = np.ones(1, dtype=dtype)
        with pytest.raises(ValueError, match="feature_weights length"):
            expected_config_cls().configure_arguments(data)

    @pytest.mark.parametrize("discretization", DISCRETIZATIONS)
    def test_valid_discretizations(
        self,
        discretization,
        make_config_data,
        expected_config_cls,
    ) -> None:
        data = make_config_data()
        data.discretization = discretization
        cfg = expected_config_cls()
        cfg.configure_arguments(data)
        assert cfg.n_features == 2

    @pytest.mark.parametrize("legacy", ["kde", "kNN"])
    def test_pre_refactor_names_rejected(
        self,
        legacy,
        make_config_data,
        expected_config_cls,
    ) -> None:
        data = make_config_data()
        data.discretization = legacy
        with pytest.raises(ValueError, match="Invalid discretization"):
            expected_config_cls().configure_arguments(data)


class TestRangeAndBandwidthValues:
    """build_cpp_config copies the fitted scale params into the right slots."""

    def test_ranges_values_match_fitted(self, make_gower) -> None:
        gower = make_gower()
        ranges = np.asarray(gower.cpp_config.ranges)
        for pos, j in enumerate(gower.numeric_indices):
            assert ranges[j] == pytest.approx(gower.numeric_ranges[pos], rel=1e-2)
        for pos, j in enumerate(gower.ratio_scale_indices):
            assert ranges[j] == pytest.approx(gower.ratio_ranges[pos], rel=1e-2)

    def test_bandwidths_values_match_fitted_silverman(self, make_gower) -> None:
        gower = make_gower(discretization="silverman")
        bandwidths = np.asarray(gower.cpp_config.bandwidths)
        for pos, j in enumerate(gower.numeric_indices):
            assert bandwidths[j] == pytest.approx(gower._h_numeric[pos], rel=1e-2)
        for pos, j in enumerate(gower.ratio_scale_indices):
            assert bandwidths[j] == pytest.approx(gower._h_ratio[pos], rel=1e-2)

    def test_bandwidths_present_with_knn(self, make_gower) -> None:
        gower = make_gower(discretization="knn")
        bandwidths = np.asarray(gower.cpp_config.bandwidths)
        scaled = sorted({*gower.numeric_indices, *gower.ratio_scale_indices})
        assert np.flatnonzero(bandwidths).tolist() == scaled


class TestFeatureWeights:
    def test_uniform_weights_all_ones(self, dtype, rng) -> None:
        df = generate_mixed_df(16, rng)
        gower = Gower(
            Config(
                feature_types=dict(MIXED_FEATURE_TYPES),
                feature_weights="uniform",
                data_type=dtype,
                categorical_ordinal_values_order={"Education": EDUCATION_LEVELS},
            ),
        ).fit(df)
        weights = np.asarray(_cpp(gower).feature_weights)
        assert weights.tolist() == [1.0] * N_MIXED_FEATURES

    def test_custom_weights_carried(self, dtype, rng) -> None:
        df = generate_mixed_df(16, rng)
        weights = {i: float(i + 2) for i in range(N_MIXED_FEATURES)}
        gower = Gower(
            Config(
                feature_types=dict(MIXED_FEATURE_TYPES),
                feature_weights=weights,
                data_type=dtype,
                categorical_ordinal_values_order={"Education": EDUCATION_LEVELS},
            ),
        ).fit(df)
        np.testing.assert_allclose(
            np.asarray(_cpp(gower).feature_weights),
            np.asarray([weights[i] for i in range(N_MIXED_FEATURES)]),
            rtol=1e-2,
        )


class TestValueOrdersHelper:
    def test_stringifies_keys_and_values(self) -> None:
        assert _value_orders({2: [1, 2.0, True], 0: ["x", "y"]}) == {
            2: ["1", "2.0", "True"],
            0: ["x", "y"],
        }

    def test_none_returns_empty(self) -> None:
        assert _value_orders(None) == {}


class TestConfigFieldsCarried:
    """Engine-relevant Config fields reach the native config via build_cpp_config."""

    def test_conditional_distances_carried(self, dtype) -> None:
        df = pd.DataFrame(
            {"n": [0.0, 1.0, 5.0], "c1": ["a", "b", "a"], "c2": ["p", "q", "p"]},
        )
        ft: dict[int | str, str] = {
            "n": "numeric",
            "c1": "categorical_nominal",
            "c2": "categorical_nominal",
        }
        plain = Gower(Config(feature_types=ft, data_type=dtype)).fit(df)
        cond = Gower(
            Config(feature_types=ft, data_type=dtype, conditional_distances=True),
        ).fit(df)
        rows = _encoded_rows(plain, df, dtype)
        assert _cpp(cond).calculate_distance(rows[0], rows[1]) == pytest.approx(1.0)
        assert _cpp(plain).calculate_distance(rows[0], rows[1]) != pytest.approx(1.0)

    def test_calc_type_podani_carried(self, dtype) -> None:
        df = pd.DataFrame({"g": ["a", "b", "b", "b", "c"]})
        order: dict[int | str, list[str]] = {"g": ["a", "b", "c"]}
        ft: dict[int | str, str] = {"g": "categorical_ordinal"}
        kauf = Gower(
            Config(
                feature_types=ft,
                categorical_ordinal_values_order=order,
                categorical_ordinal_calculation_type="kaufman",
                data_type=dtype,
            ),
        ).fit(df)
        pod = Gower(
            Config(
                feature_types=ft,
                categorical_ordinal_values_order=order,
                categorical_ordinal_calculation_type="podani",
                data_type=dtype,
            ),
        ).fit(df)
        rows = _encoded_rows(kauf, df, dtype)
        assert _cpp(kauf).calculate_distance(rows[0], rows[1]) == pytest.approx(0.5)
        assert _cpp(pod).calculate_distance(rows[0], rows[1]) == pytest.approx(0.0)

    def test_missing_strategy_carried(self, dtype) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "x", "z"]})
        ft: dict[int | str, str] = {"a": "numeric", "b": "categorical_nominal"}
        ignore = Gower(
            Config(feature_types=ft, data_type=dtype, missing_strategy="ignore"),
        ).fit(df)
        max_dist = Gower(
            Config(feature_types=ft, data_type=dtype, missing_strategy="max_dist"),
        ).fit(df)
        rows = _encoded_rows(ignore, df, dtype)
        x = rows[0].copy()
        x[0] = np.nan
        assert _cpp(ignore).calculate_distance(x, rows[1]) == pytest.approx(0.0)
        assert _cpp(max_dist).calculate_distance(x, rows[1]) == pytest.approx(0.5)


class TestDiscretizationConfigCarried:
    """Discretization fields introduced in the scale_window refactor reach the engine."""

    def _fit(self, df: pd.DataFrame, dtype, **cfg_kwargs: Any) -> Gower:
        cfg = Config(
            feature_types=dict(MIXED_FEATURE_TYPES),
            data_type=dtype,
            categorical_ordinal_values_order={"Education": EDUCATION_LEVELS},
            **cfg_kwargs,
        )
        return Gower(cfg).fit(df)

    def test_silverman_constant_scales_bandwidths(self, dtype, rng) -> None:
        df = generate_mixed_df(64, rng)
        base = self._fit(df, dtype, discretization="silverman")
        doubled = self._fit(
            df,
            dtype,
            discretization="silverman",
            silverman_constant=2.12,
        )
        b_base = np.asarray(_cpp(base).bandwidths, dtype=np.float64)
        b_doubled = np.asarray(_cpp(doubled).bandwidths, dtype=np.float64)
        nonzero = b_base != 0
        assert nonzero.any()
        np.testing.assert_allclose(b_doubled[nonzero] / b_base[nonzero], 2.0, rtol=1e-2)

    def test_k_neighbors_changes_knn_bandwidths(self, dtype, rng) -> None:
        df = generate_mixed_df(64, rng)
        small_k = self._fit(df, dtype, discretization="knn", k_neighbors=2)
        large_k = self._fit(df, dtype, discretization="knn", k_neighbors=32)
        b_small = np.asarray(_cpp(small_k).bandwidths, dtype=np.float64)
        b_large = np.asarray(_cpp(large_k).bandwidths, dtype=np.float64)
        nonzero = b_large != 0
        assert nonzero.any()
        assert np.all(b_large[nonzero] > b_small[nonzero])

    def test_knn_default_k_matches_sqrt_n(self, dtype, rng) -> None:
        df = generate_mixed_df(64, rng)
        default_k = self._fit(df, dtype, discretization="knn")
        explicit_k = self._fit(df, dtype, discretization="knn", k_neighbors=8)
        np.testing.assert_allclose(
            np.asarray(_cpp(default_k).bandwidths, dtype=np.float64),
            np.asarray(_cpp(explicit_k).bandwidths, dtype=np.float64),
            rtol=1e-6,
        )


class TestGroupedKernelSemantics:
    """Corners the type-grouped kernel has to preserve."""

    def test_raise_error_wins_over_conditional_short_circuit(self, dtype) -> None:
        df = pd.DataFrame(
            {
                "n": [0.0, 1.0, 5.0, 9.0],
                "c1": ["a", "b", "a", "b"],
                "c2": ["p", "q", "p", "q"],
            },
        )
        ft: dict[int | str, str] = {
            "n": "numeric",
            "c1": "categorical_nominal",
            "c2": "categorical_nominal",
        }
        gower = Gower(
            Config(
                feature_types=ft,
                data_type=dtype,
                conditional_distances=True,
                missing_strategy="raise_error",
            ),
        ).fit(df)
        rows = _encoded_rows(gower, df, dtype)
        x = rows[0].copy()
        x[0] = np.nan
        with pytest.raises(ValueError, match="Missing values detected"):
            _cpp(gower).calculate_distance(x, rows[1])

    @pytest.mark.parametrize("strategy", ["ignore", "max_dist"])
    def test_joint_absence_excluded_under_every_missing_strategy(
        self,
        strategy,
        dtype,
    ) -> None:
        df = pd.DataFrame({"a": [1.0, 0.0, 1.0, 0.0], "b": [0, 1, 0, 1]})
        ft: dict[int | str, str] = {"a": "numeric", "b": "binary_asymmetric"}
        gower = Gower(
            Config(feature_types=ft, data_type=dtype, missing_strategy=strategy),
        ).fit(df)
        rows = _encoded_rows(gower, df, dtype)
        both_absent = rows[0].copy()
        both_absent[1] = 0.0
        other_absent = rows[0].copy()
        other_absent[1] = 0.0
        other_absent[0] = rows[2][0]
        numeric_only = abs(float(both_absent[0]) - float(other_absent[0])) / 1.0
        assert _cpp(gower).calculate_distance(
            both_absent,
            other_absent,
        ) == pytest.approx(numeric_only, abs=1e-3)


class TestBlockedMatrixMatchesScalarKernel:
    """matrix() runs a blocked column-major kernel, calculate_distance a scalar one."""

    ORDER: ClassVar[dict[int | str, list[str]]] = {"Education": EDUCATION_LEVELS}

    @pytest.mark.parametrize(
        "extra",
        [
            pytest.param({}, id="defaults"),
            pytest.param({"feature_weights": "uniform"}, id="uniform_weights"),
            pytest.param(
                {"feature_weights": {i: float(i + 1) for i in range(6)}},
                id="custom_weights",
            ),
            pytest.param({"missing_strategy": "max_dist"}, id="max_dist"),
            pytest.param({"discretization": "silverman"}, id="silverman"),
            pytest.param({"discretization": "knn"}, id="knn"),
            pytest.param(
                {"categorical_ordinal_calculation_type": "podani"},
                id="podani",
            ),
            pytest.param({"conditional_distances": True}, id="conditional"),
        ],
    )
    def test_matrix_matches_pairwise_distance(self, extra, dtype, rng) -> None:
        df = generate_mixed_df(48, rng)
        gower = Gower(
            Config(
                feature_types=dict(MIXED_FEATURE_TYPES),
                data_type=dtype,
                out_of_range="clip",
                categorical_ordinal_values_order=dict(self.ORDER),
                **extra,
            ),
        ).fit(df)
        rows = _encoded_rows(gower, df, dtype)
        matrix = np.asarray(gower.matrix(rows))
        calc = _cpp(gower).calculate_distance
        expected = np.array(
            [[calc(a, b) for b in rows] for a in rows],
            dtype=np.float64,
        )
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_array_equal(matrix.astype(np.float64), expected)

    def test_missing_values_match_pairwise_distance(self, dtype, rng) -> None:
        df = generate_mixed_df(40, rng)
        df.loc[df.index[::3], "Age"] = np.nan
        df.loc[df.index[::5], "Salary"] = np.nan
        gower = Gower(
            Config(
                feature_types=dict(MIXED_FEATURE_TYPES),
                data_type=dtype,
                out_of_range="clip",
                missing_strategy="ignore",
                categorical_ordinal_values_order=dict(self.ORDER),
            ),
        ).fit(df)
        rows = _encoded_rows(gower, df, dtype)
        matrix = np.asarray(gower.matrix(rows))
        calc = _cpp(gower).calculate_distance
        expected = np.array(
            [[calc(a, b) for b in rows] for a in rows],
            dtype=np.float64,
        )
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_array_equal(matrix.astype(np.float64), expected)

    def test_live_podani_matches_pairwise_distance(self, dtype) -> None:
        levels = ["a", "b", "c", "d", "e"]
        g = ["a", *["b"] * 3, *["c"] * 3, *["d"] * 3, "e"]
        df = pd.DataFrame({"g": g, "n": np.arange(len(g), dtype=float)})

        def fit(calc_type: Literal["kaufman", "podani"]) -> Gower:
            return Gower(
                Config(
                    feature_types={"g": "categorical_ordinal", "n": "numeric"},
                    categorical_ordinal_values_order={"g": levels},
                    categorical_ordinal_calculation_type=calc_type,
                    data_type=dtype,
                    out_of_range="clip",
                ),
            ).fit(df)

        gower = fit("podani")
        rows = _encoded_rows(gower, df, dtype)
        matrix = np.asarray(gower.matrix(rows))
        calc = _cpp(gower).calculate_distance
        expected = np.array(
            [[calc(a, b) for b in rows] for a in rows],
            dtype=np.float64,
        )
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_array_equal(matrix.astype(np.float64), expected)

        kaufman = np.asarray(fit("kaufman").matrix(rows))
        assert not np.array_equal(matrix, kaufman)

    def test_cross_matrix_matches_pairwise_distance(self, dtype, rng) -> None:
        df = generate_mixed_df(60, rng)
        gower = Gower(
            Config(
                feature_types=dict(MIXED_FEATURE_TYPES),
                data_type=dtype,
                out_of_range="clip",
                categorical_ordinal_values_order=dict(self.ORDER),
            ),
        ).fit(df)
        rows = _encoded_rows(gower, df, dtype)
        x_rows, y_rows = rows[:20], rows[20:]
        matrix = np.asarray(gower.matrix(x_rows, Y=y_rows))
        calc = _cpp(gower).calculate_distance
        expected = np.array(
            [[calc(a, b) for b in y_rows] for a in x_rows],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(matrix.astype(np.float64), expected)

    NO_ASYM: ClassVar[dict[int | str, str]] = {
        k: v for k, v in MIXED_FEATURE_TYPES.items() if v != "binary_asymmetric"
    }

    @pytest.mark.parametrize("poison", [False, True], ids=["nan_free", "with_nan"])
    def test_constant_denominator_matches_pairwise_distance(
        self,
        poison,
        dtype,
        rng,
    ) -> None:
        df = generate_mixed_df(48, rng)[list(self.NO_ASYM)]
        gower = Gower(
            Config(
                feature_types=dict(self.NO_ASYM),
                data_type=dtype,
                out_of_range="clip",
                feature_weights={i: float(i + 1) for i in range(len(self.NO_ASYM))},
                categorical_ordinal_values_order=dict(self.ORDER),
            ),
        ).fit(df)
        rows = np.array(_encoded_rows(gower, df, dtype), copy=True)
        if poison:
            rows[0, 0] = np.nan
        matrix = np.asarray(gower.matrix(rows))
        calc = _cpp(gower).calculate_distance
        expected = np.array(
            [[calc(a, b) for b in rows] for a in rows],
            dtype=np.float64,
        )
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_array_equal(matrix.astype(np.float64), expected)

    def test_all_zero_weights_stay_undefined(self, dtype, rng) -> None:
        df = generate_mixed_df(16, rng)[list(self.NO_ASYM)]
        gower = Gower(
            Config(
                feature_types=dict(self.NO_ASYM),
                data_type=dtype,
                out_of_range="clip",
                feature_weights=dict.fromkeys(range(len(self.NO_ASYM)), 0.0),
                categorical_ordinal_values_order=dict(self.ORDER),
            ),
        ).fit(df)
        rows = _encoded_rows(gower, df, dtype)
        matrix = np.asarray(gower.matrix(rows))
        off_diagonal = ~np.eye(len(rows), dtype=bool)
        assert np.isnan(matrix[off_diagonal]).all()

    def test_similarity_matches_pairwise_distance(self, dtype, rng) -> None:
        df = generate_mixed_df(40, rng)
        gower = Gower(
            Config(
                feature_types=dict(MIXED_FEATURE_TYPES),
                data_type=dtype,
                out_of_range="clip",
                categorical_ordinal_values_order=dict(self.ORDER),
            ),
        ).fit(df)
        rows = _encoded_rows(gower, df, dtype)
        matrix = np.asarray(gower.matrix(rows, matrix_type="similarity"))
        calc = _cpp(gower).calculate_distance
        expected = np.array(
            [[dtype(1.0 - calc(a, b)) for b in rows] for a in rows],
            dtype=np.float64,
        )
        np.fill_diagonal(expected, 1.0)
        np.testing.assert_array_equal(matrix.astype(np.float64), expected)


class TestPodaniFallbackWarningAtFit:
    def test_fit_emits_warning_when_podani_degenerates(self, dtype) -> None:
        df = pd.DataFrame({"g": ["low", "low"]})
        cfg = Config(
            feature_types={"g": "categorical_ordinal"},
            categorical_ordinal_values_order={"g": ["low"]},
            categorical_ordinal_calculation_type="podani",
            data_type=dtype,
        )
        with pytest.warns(UserWarning, match="Podani denominator"):
            Gower(cfg).fit(df)


class TestBuilderEdgeCases:
    def test_all_categorical_ranges_all_zero(self, dtype) -> None:
        df = pd.DataFrame({"a": ["x", "y", "x"], "b": ["p", "q", "p"]})
        ft: dict[int | str, str] = {
            "a": "categorical_nominal",
            "b": "categorical_nominal",
        }
        gower = Gower(Config(feature_types=ft, data_type=dtype)).fit(df)
        assert not np.any(np.asarray(_cpp(gower).ranges))
        assert _cpp(gower).n_features == 2

    def test_builds_without_ordinal_feature(self, dtype) -> None:
        df = pd.DataFrame({"n": [1.0, 2.0], "c": ["a", "b"]})
        ft: dict[int | str, str] = {"n": "numeric", "c": "categorical_nominal"}
        gower = Gower(Config(feature_types=ft, data_type=dtype)).fit(df)
        assert _cpp(gower).n_features == 2
