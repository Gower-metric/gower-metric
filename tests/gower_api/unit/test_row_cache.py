# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

"""Behaviour tests for the per-record encode cache used by ``Gower.__call__``."""

import pickle
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from gower_metric import Config, Gower
from gower_metric.core.metric import _NUMERIC_DTYPE_KINDS
from gower_metric.utils.matrix.distance import calculate_matrix
from gower_metric.utils.row_cache import RowEncodeCache, _record_key
from gower_metric.utils.to_array import to_array

FEATURE_TYPES: dict[int | str, str] = {
    0: "numeric",
    1: "categorical_nominal",
    2: "binary_symmetric",
}


def _data(n: int = 12) -> npt.NDArray[np.generic]:
    rng = np.random.default_rng(seed=7)
    arr = np.empty((n, 3), dtype=object)
    arr[:, 0] = rng.random(n)
    arr[:, 1] = rng.choice(list("abc"), n)
    arr[:, 2] = rng.integers(0, 2, n)
    return arr


def _object_array(values: list[object]) -> npt.NDArray[np.object_]:
    """Build a record array preserving mixed types."""
    return np.array(values, dtype=object)


def _fitted(data: npt.NDArray[np.generic]) -> Gower:
    return Gower(Config(feature_types=FEATURE_TYPES)).fit(data)


class TestCacheIsInvisible:
    def test_repeated_calls_are_stable(self) -> None:
        data = _data()
        gower = _fitted(data)

        first = float(gower(data[0], data[1]))
        again = float(gower(data[0], data[1]))

        assert first == again

    def test_warm_cache_matches_cold_instance(self) -> None:
        data = _data()
        warm = _fitted(data)
        cold = _fitted(data)

        for i in range(len(data)):
            warm(data[i], data[(i + 1) % len(data)])

        for i in range(len(data)):
            for j in range(len(data)):
                assert float(warm(data[i], data[j])) == float(
                    cold(data[i], data[j]),
                )

    def test_pairwise_loop_matches_matrix(self) -> None:
        data = _data()
        gower = _fitted(data)

        n = len(data)
        looped = np.array(
            [[float(gower(data[i], data[j])) for j in range(n)] for i in range(n)],
        )

        np.testing.assert_array_equal(looped, calculate_matrix(gower, data))

    def test_records_with_nan_are_supported(self) -> None:
        data = _data()
        gower = _fitted(data)
        row = data[0].copy()
        row[0] = np.nan

        first = float(gower(row, data[1]))
        again = float(gower(row, data[1]))

        assert first == again
        assert not np.isnan(first)

    def test_self_distance_is_zero(self) -> None:
        """Both keys are identical, exercising the duplicate-key store path."""
        data = _data()
        gower = _fitted(data)

        assert float(gower(data[0], data[0])) == pytest.approx(0.0)


class TestFastPathsStayEquivalent:
    """The hot path trades library calls for cheaper equivalents."""

    @pytest.mark.parametrize(
        "dtype",
        [
            np.int32,
            np.uint8,
            np.float16,
            np.float64,
            np.complex128,
            np.bool_,
            object,
            np.str_,
            np.bytes_,
            "datetime64[s]",
            "timedelta64[s]",
            "V8",
        ],
    )
    def test_numeric_kinds_match_issubdtype(self, dtype: Any) -> None:
        """``timedelta64`` is the trap: NumPy files it under ``signedinteger``."""
        resolved = np.dtype(dtype)

        assert (resolved.kind in _NUMERIC_DTYPE_KINDS) == bool(
            np.issubdtype(resolved, np.number),
        )

    @pytest.mark.parametrize("wrap", [list, pd.Series, _object_array])
    def test_record_key_matches_the_converted_record(
        self,
        wrap: Callable[[list[Any]], Any],
    ) -> None:
        """The key is built from the raw record, bypassing ``to_array``."""
        values = [1.0, "a", 0]
        record = wrap(values)

        via_conversion = tuple(to_array(record).reshape(1, -1)[0].tolist())

        assert _record_key(record) == via_conversion

    @pytest.mark.parametrize("wrap", [list, pd.Series, _object_array])
    def test_input_container_does_not_change_the_distance(
        self,
        wrap: Callable[[list[Any]], Any],
    ) -> None:
        data = _data()
        gower = _fitted(data)
        expected = float(gower(data[0], data[1]))

        wrapped_a = wrap(list(data[0]))
        wrapped_b = wrap(list(data[1]))

        assert float(gower(wrapped_a, wrapped_b)) == expected


class TestCacheLifecycle:
    def test_cache_populates(self) -> None:
        data = _data()
        gower = _fitted(data)

        assert len(gower._row_cache) == 0
        gower(data[0], data[1])
        assert len(gower._row_cache) == 2

    def test_refit_clears_cache(self) -> None:
        data = _data()
        gower = _fitted(data)
        gower(data[0], data[1])
        assert len(gower._row_cache) > 0

        gower.fit(data)

        assert len(gower._row_cache) == 0

    def test_refit_changes_results_rather_than_serving_stale_rows(self) -> None:
        """A refit on different data must not be masked by cached encodings."""
        train = _data()
        gower = _fitted(train)
        probe_a, probe_b = train[0], train[1]
        before = float(gower(probe_a, probe_b))

        widened = train.copy()
        widened[:, 0] = widened[:, 0].astype(float) * 100.0
        gower.fit(widened)

        after = float(gower(probe_a, probe_b))

        assert before != after

    def test_pickle_drops_cache_but_keeps_results(self) -> None:
        data = _data()
        gower = _fitted(data)
        expected = float(gower(data[0], data[1]))

        restored = pickle.loads(pickle.dumps(gower))  # noqa: S301

        assert len(restored._row_cache) == 0
        assert float(restored(data[0], data[1])) == expected

    @pytest.mark.parametrize("missing", ["cpp_config", "_row_cache"])
    def test_state_predating_a_derived_field_is_restored(self, missing: str) -> None:
        """A pickle from before a derived field existed must still be usable."""
        data = _data()
        gower = _fitted(data)
        expected = float(gower(data[0], data[1]))

        legacy_state = {k: v for k, v in gower.__dict__.items() if k != missing}
        restored = Gower.__new__(Gower)
        restored.__setstate__(legacy_state)

        assert float(restored(data[0], data[1])) == expected

    def test_cache_is_bounded(self) -> None:
        data = _data(n=40)
        gower = _fitted(data)
        gower._row_cache.maxsize = 6

        for i in range(len(data)):
            gower(data[i], data[(i + 1) % len(data)])

        assert len(gower._row_cache) == 6


class TestOutOfRangeStillReported:
    def test_error_fires_on_every_call(self) -> None:
        """An uncacheable outcome must not be silently swallowed on retry."""
        train = np.array([[1.0, "a", 0], [5.0, "b", 1]], dtype=object)
        cfg = Config(feature_types=FEATURE_TYPES, out_of_range="error")
        gower = Gower(cfg).fit(train)
        offending = np.array([99.0, "a", 0], dtype=object)

        for _ in range(2):
            with pytest.raises(ValueError, match=r"Out-of-range"):
                gower(offending, train[0])

    def test_matrix_reports_out_of_range_once(self) -> None:
        """matrix() checks up front, then suppresses the per-transform recheck."""
        train = np.array([[1.0, "a", 0], [5.0, "b", 1]], dtype=object)
        probe = np.array([[1.0, "a", 0], [5.0, "b", 1], [99.0, "a", 0]], dtype=object)
        cfg = Config(feature_types=FEATURE_TYPES, out_of_range="warning")
        gower = Gower(cfg).fit(train)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calculate_matrix(gower, probe)

        oor = [w for w in caught if "Out-of-range" in str(w.message)]
        assert len(oor) == 1

    def test_matrix_restores_skip_oor_afterwards(self) -> None:
        train = np.array([[1.0, "a", 0], [5.0, "b", 1]], dtype=object)
        cfg = Config(feature_types=FEATURE_TYPES, out_of_range="warning")
        gower = Gower(cfg).fit(train)
        before = gower.skip_oor

        calculate_matrix(gower, train)

        assert gower.skip_oor == before


class TestUncacheableRecords:
    def test_unhashable_record_is_encoded_every_time(self) -> None:
        """A list inside a record cannot be keyed; that must not raise."""
        cache = RowEncodeCache()
        calls = []

        def transform(rows: npt.NDArray[np.object_]) -> npt.NDArray[np.floating]:
            calls.append(rows.shape)
            return np.zeros((rows.shape[0], rows.shape[1]), dtype=np.float64)

        record = np.array([1.0, ["unhashable"], 0], dtype=object)

        for _ in range(2):
            cache.encode_pair(record, record, transform, np.float64)

        assert len(calls) == 2
        assert len(cache) == 0
