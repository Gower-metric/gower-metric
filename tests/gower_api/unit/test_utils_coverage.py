# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

"""Tests for utility functions — aux, cat_ord_ut, ranges, silverman, knn_bandwidth, categorical_ut."""

import numpy as np
import pandas as pd
import pytest

from gower_metric.utils.auxiliary import all_ones_off_diagonal
from gower_metric.utils.cat_ord_ut import map_ordered_values
from gower_metric.utils.discretization_types import knn, silverman
from gower_metric.utils.ranges import (
    check_out_of_range,
    get_numeric_bounds,
    scale_span,
)
from gower_metric.utils.to_array import to_array


class TestToArray:
    def test_list_input(self) -> None:
        result = to_array([1, 2, "a"])
        assert isinstance(result, np.ndarray)
        assert result.dtype == object


class TestAllOnesOffDiagonal:
    def test_with_dataframe(self) -> None:
        arr = np.ones((3, 3))
        np.fill_diagonal(arr, 0.0)
        df = pd.DataFrame(arr)
        assert all_ones_off_diagonal(df) is True

    def test_not_all_ones(self) -> None:
        arr = np.array([[0.0, 0.5], [0.5, 0.0]])
        assert all_ones_off_diagonal(arr) is False

    def test_with_ndarray(self) -> None:
        arr = np.ones((2, 2))
        np.fill_diagonal(arr, 0.0)
        assert all_ones_off_diagonal(arr) is True

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected DataFrame or ndarray"):
            all_ones_off_diagonal([[1, 1], [1, 1]])  # type: ignore[arg-type]


class TestScaleMethod:
    def test_empty_array_returns_zero(self) -> None:
        assert scale_span(np.array([]), "range") == 0.0

    def test_iqr_method(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = scale_span(arr, "iqr")
        assert result > 0

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            scale_span(np.array([1.0, 2.0]), "bad_method")

    def test_constant_values_returns_zero(self) -> None:
        assert scale_span(np.array([5.0, 5.0, 5.0]), "range") == 0.0


class TestSilvermanBandwidth:
    def test_less_than_two_samples_returns_zero(self) -> None:
        assert silverman.bandwidth(np.array([1.0])) == 0.0

    def test_empty_after_nan_removal(self) -> None:
        assert silverman.bandwidth(np.array([np.nan])) == 0.0

    def test_normal_data(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        h = silverman.bandwidth(data)
        assert h > 0


class TestKnnBandwidth:
    def test_single_point_returns_zero(self) -> None:
        assert knn.bandwidth(np.array([5.0])) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert knn.bandwidth(np.array([])) == 0.0

    def test_normal_data(self) -> None:
        data = np.arange(100, dtype=float)
        h = knn.bandwidth(data)
        assert h > 0

    def test_with_explicit_k(self) -> None:
        data = np.arange(20, dtype=float)
        h = knn.bandwidth(data, k=3)
        assert h > 0


class TestNumericBoundsEdgeCases:
    def test_all_nan_column_yields_nan_bounds(self) -> None:
        """An unusable column must not poison the fitted bounds with a number."""
        X = np.array([[np.nan, 1.0], [np.nan, 2.0]], dtype=object)

        mins, maxs = get_numeric_bounds(X, [0, 1])

        assert np.isnan(mins[0])
        assert np.isnan(maxs[0])
        assert (mins[1], maxs[1]) == (1.0, 2.0)


class TestCheckOutOfRangeEdgeCases:
    def test_empty_input_reports_nothing(self) -> None:
        """A zero-row reduction has no identity, so it is short-circuited."""
        X = np.empty((0, 2), dtype=object)

        assert (
            check_out_of_range(X, [0], np.array([0.0]), np.array([1.0]), "numeric")
            == []
        )


class TestMapOrderedValuesEdgeCases:
    def test_empty_order_yields_no_ranks(self) -> None:
        mapping, min_rank, max_rank = map_ordered_values([])

        assert mapping == {}
        assert min_rank is None
        assert max_rank is None
