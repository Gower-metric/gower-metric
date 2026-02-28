"""Tests for missing.py — first_not_missing and apply_missing_strategy branches."""

import numpy as np
import pytest

from gower_metric.utils.missing import apply_missing_strategy, first_not_missing


class TestFirstNotMissing:
    def test_returns_first_non_missing(self) -> None:
        assert first_not_missing([None, float("nan"), 42, 5]) == 42

    def test_all_missing_returns_none(self) -> None:
        assert first_not_missing([None, float("nan"), None]) is None

    def test_empty_sequence_returns_none(self) -> None:
        assert first_not_missing([]) is None

    def test_first_element_valid(self) -> None:
        assert first_not_missing([7, None, 3]) == 7


class TestApplyMissingStrategy:
    def _make_data(self) -> tuple[np.ndarray, np.ndarray]:
        diff = np.array([[0.5, 0.3], [0.7, 0.1]])
        present = np.array([[True, False], [False, True]])
        return diff, present

    def test_ignore_sets_missing_to_zero(self) -> None:
        diff, present = self._make_data()
        result_diff, count = apply_missing_strategy(diff, present, "ignore")
        assert result_diff[0, 1] == 0.0
        assert result_diff[1, 0] == 0.0
        assert count[0, 0] == 1
        assert count[0, 1] == 0

    def test_max_dist_sets_missing_to_one(self) -> None:
        diff, present = self._make_data()
        result_diff, _count = apply_missing_strategy(diff, present, "max_dist")
        assert result_diff[0, 1] == 1.0
        assert result_diff[1, 0] == 1.0

    def test_raise_error_with_missing_raises(self) -> None:
        diff, present = self._make_data()
        with pytest.raises(ValueError):
            apply_missing_strategy(diff, present, "raise_error")

    def test_raise_error_all_present_passes(self) -> None:
        diff = np.array([[0.5, 0.3], [0.7, 0.1]])
        present = np.ones((2, 2), dtype=bool)
        _result_diff, count = apply_missing_strategy(diff, present, "raise_error")
        np.testing.assert_array_equal(count, np.ones((2, 2), dtype=int))

    def test_unknown_method_raises(self) -> None:
        diff = np.array([[0.5]])
        present = np.array([[True]])
        with pytest.raises(ValueError, match="Unknown nan_method"):
            apply_missing_strategy(diff, present, "bad_method")
