"""Shared fixtures for the native (C++/nanobind) test suite."""

import zlib
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from gower_metric import Config, Gower
from gower_metric.cpp import (
    CppConfig,
    CppConfigData,
    CppConfigDataF,
    CppConfigDataH,
    CppConfigF,
    CppConfigH,
)
from tests.conftest import EDUCATION_LEVELS, NUMPY_NUMERIC_TYPES, generate_mixed_df

DTYPE_TO_CONFIG: dict[type[np.floating], type] = {
    np.float16: CppConfigH,
    np.float32: CppConfigF,
    np.float64: CppConfig,
}
DTYPE_TO_CONFIG_DATA: dict[type[np.floating], type] = {
    np.float16: CppConfigDataH,
    np.float32: CppConfigDataF,
    np.float64: CppConfigData,
}

FEATURE_TYPES = (
    "numeric",
    "ratio_scale_interval",
    "categorical_nominal",
    "categorical_ordinal",
    "binary_symmetric",
    "binary_asymmetric",
)
MISSING_STRATEGIES = ("ignore", "max_dist", "raise_error")
SCALE_METHODS = ("range", "iqr")
DISCRETIZATIONS = ("", "silverman", "knn")

MIXED_FEATURE_TYPES: dict[int | str, str] = {
    "Age": "numeric",
    "Salary": "ratio_scale_interval",
    "Have_children": "binary_symmetric",
    "Is_smoking": "binary_asymmetric",
    "Birth": "categorical_nominal",
    "Education": "categorical_ordinal",
}


@pytest.fixture(params=NUMPY_NUMERIC_TYPES, ids=lambda dt: dt.__name__)
def dtype(request: pytest.FixtureRequest) -> type[np.floating]:
    """Run the test once per supported float dtype (float16/float32/float64)."""
    return cast("type[np.floating]", request.param)


@pytest.fixture
def expected_config_cls(dtype: type[np.floating]) -> type:
    """Native CppConfig class build_cpp_config should pick for ``dtype``."""
    return DTYPE_TO_CONFIG[dtype]


@pytest.fixture
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """Per-test seeded NumPy RNG."""
    return np.random.default_rng(zlib.crc32(request.node.name.encode()))


@pytest.fixture
def make_gower(
    dtype: type[np.floating],
    rng: np.random.Generator,
) -> Callable[..., Gower]:
    """Return a factory building a real Gower instance."""

    def _make(*, n: int = 32, discretization=None) -> Gower:
        df = generate_mixed_df(n, rng)
        cfg = Config(
            feature_types=dict(MIXED_FEATURE_TYPES),
            feature_weights={i: float(i + 1) for i in range(len(MIXED_FEATURE_TYPES))},
            data_type=dtype,
            categorical_ordinal_values_order={"Education": EDUCATION_LEVELS},
            discretization=discretization,
        )
        return Gower(cfg).fit(df)

    return _make


@pytest.fixture
def make_config_data(dtype: type[np.floating]) -> Callable[..., Any]:
    """Return a factory for a valid native CppConfigData to populate by hand."""

    def _make(
        feature_types: tuple[str, ...] = ("numeric", "numeric"),
        *,
        calculation_type: str = "kaufman",
        ordinal_counts: dict[int, list[float]] | None = None,
        missing_strategy: str = "ignore",
        conditional_distances: bool = False,
        conditional_distances_threshold_coeff: int = 1,
        feature_weights: list[float] | None = None,
    ) -> Any:
        data = DTYPE_TO_CONFIG_DATA[dtype]()
        n = len(feature_types)
        data.feature_types = list(feature_types)
        data.feature_weights = np.asarray(
            [1.0] * n if feature_weights is None else feature_weights,
            dtype=dtype,
        )
        data.ranges = np.ones(n, dtype=dtype)
        data.bandwidths = np.zeros(n, dtype=dtype)
        data.missing_strategy = missing_strategy
        data.scale_method = "range"
        data.categorical_ordinal_calculation_type = calculation_type
        data.conditional_distances = conditional_distances
        data.conditional_distances_threshold_coeff = (
            conditional_distances_threshold_coeff
        )
        data.handle_unseen_binary_asymmetric = "error"
        data.handle_unseen_binary_symmetric = "error"
        data.handle_unseen_categorical_nominal = "error"
        data.handle_unseen_categorical_ordinal = "error"
        data.out_of_range = "error"
        if ordinal_counts is not None:
            data.ordinal_counts = ordinal_counts
        return data

    return _make
