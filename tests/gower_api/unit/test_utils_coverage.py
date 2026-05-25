"""Tests for utility functions — aux, cat_ord_ut, ranges, silverman, knn_bandwidth, categorical_ut."""

import numpy as np
import pandas as pd
import pytest

from gower_metric.utils.auxiliary import all_ones_off_diagonal
from gower_metric.utils.kde_types.silverman import silverman_bandwidth
from gower_metric.utils.knn_bandwidth import knn_bandwidth
from gower_metric.utils.ranges import scale_method
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
        assert scale_method(np.array([]), "range") == 0.0

    def test_iqr_method(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = scale_method(arr, "iqr")
        assert result > 0

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            scale_method(np.array([1.0, 2.0]), "bad_method")

    def test_constant_values_returns_zero(self) -> None:
        assert scale_method(np.array([5.0, 5.0, 5.0]), "range") == 0.0


class TestSilvermanBandwidth:
    def test_less_than_two_samples_returns_zero(self) -> None:
        assert silverman_bandwidth(np.array([1.0])) == 0.0

    def test_empty_after_nan_removal(self) -> None:
        assert silverman_bandwidth(np.array([np.nan])) == 0.0

    def test_normal_data(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        h = silverman_bandwidth(data)
        assert h > 0


class TestKnnBandwidth:
    def test_single_point_returns_zero(self) -> None:
        assert knn_bandwidth(np.array([5.0])) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert knn_bandwidth(np.array([])) == 0.0

    def test_normal_data(self) -> None:
        data = np.arange(100, dtype=float)
        h = knn_bandwidth(data)
        assert h > 0

    def test_with_explicit_k(self) -> None:
        data = np.arange(20, dtype=float)
        h = knn_bandwidth(data, k=3)
        assert h > 0
