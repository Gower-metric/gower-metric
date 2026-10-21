# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

"""Differential tests for the fast category encoder."""

import pickle
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.preprocessing import OrdinalEncoder

from gower_metric.utils.transforms.encoding import (
    MAP_THRESHOLD,
    _code_mapping,
    encode_categories,
)

CATEGORY_SETS: dict[str, tuple[list[Any], list[Any]]] = {
    "text": (list("abcde"), list("acez")),
    "integers": ([1, 2, 3, 10], [1, 10, 99]),
    "floats": ([1.5, 2.5, 3.5], [1.5, 9.5]),
    "single_category": (["x"], ["x", "y"]),
    "all_unknown": (["a", "b"], ["y", "z"]),
}

DTYPES = [np.float16, np.float32, np.float64]


def _object(values: list[object]) -> npt.NDArray[np.object_]:
    return np.array(values, dtype=object)


def _fit(
    categories: list[Any],
    dtype: type[np.floating],
    *,
    strict: bool,
    explicit_order: bool = False,
) -> OrdinalEncoder:
    """Fit an encoder the way the library's fit helpers do."""
    kwargs: dict[str, Any] = (
        {"handle_unknown": "error"}
        if strict
        else {"handle_unknown": "use_encoded_value", "unknown_value": np.nan}
    )
    if explicit_order:
        kwargs["categories"] = [categories[::-1]]
    enc = OrdinalEncoder(dtype=dtype, **kwargs)  # type: ignore[no-untyped-call]
    enc.fit(_object(categories).reshape(-1, 1))
    return enc


class TestMatchesScikitLearn:
    @pytest.mark.parametrize("case", CATEGORY_SETS)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("explicit_order", [False, True])
    def test_codes_match(
        self,
        case: str,
        dtype: type[np.floating],
        explicit_order: bool,
    ) -> None:
        categories, probe_values = CATEGORY_SETS[case]
        enc = _fit(categories, dtype, strict=False, explicit_order=explicit_order)
        probe = _object(probe_values)

        expected = enc.transform(probe.reshape(-1, 1)).astype(dtype).ravel()  # type: ignore[no-untyped-call]

        np.testing.assert_array_equal(
            encode_categories(probe, enc, dtype),
            expected,
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_output_dtype_matches(self, dtype: type[np.floating]) -> None:
        enc = _fit(list("abc"), dtype, strict=False)
        probe = _object(list("abc"))

        assert encode_categories(probe, enc, dtype).dtype == dtype

    def test_unsorted_user_order_is_respected(self) -> None:
        """Ordinal columns carry the user's order, not a sorted one."""
        enc = _fit(["low", "medium", "high"], np.float64, strict=False)
        enc.categories = [["low", "medium", "high"]]
        enc.fit(_object(["low", "medium", "high"]).reshape(-1, 1))

        codes = encode_categories(_object(["low", "medium", "high"]), enc, np.float64)

        np.testing.assert_array_equal(codes, [0.0, 1.0, 2.0])


class TestBothStrategiesAgree:
    """The row count picks between a Python lookup and ``Series.map``."""

    @pytest.mark.parametrize("count", [1, MAP_THRESHOLD - 1, MAP_THRESHOLD + 1])
    def test_result_is_independent_of_the_chosen_strategy(self, count: int) -> None:
        categories = list("abcde")
        enc = _fit(categories, np.float64, strict=False)
        probe = _object([*(categories * count)[:count]])

        expected = enc.transform(probe.reshape(-1, 1)).astype(np.float64).ravel()  # type: ignore[no-untyped-call]

        np.testing.assert_array_equal(
            encode_categories(probe, enc, np.float64),
            expected,
        )

    def test_unknowns_survive_the_large_branch(self) -> None:
        enc = _fit(list("abc"), np.float64, strict=False)
        probe = _object(["a", "zzz"] * MAP_THRESHOLD)

        codes = encode_categories(probe, enc, np.float64)

        assert np.isnan(codes[1::2]).all()
        assert not np.isnan(codes[0::2]).any()


class TestStrictEncoderRaises:
    @pytest.mark.parametrize("explicit_order", [False, True])
    def test_unknown_value_raises(self, explicit_order: bool) -> None:
        enc = _fit(list("abc"), np.float64, strict=True, explicit_order=explicit_order)

        with pytest.raises(ValueError, match=r"Found unknown categories") as excinfo:
            encode_categories(_object(["a", "zzz"]), enc, np.float64)

        assert "zzz" in str(excinfo.value)

    def test_known_values_do_not_raise(self) -> None:
        enc = _fit(list("abc"), np.float64, strict=True)

        codes = encode_categories(_object(["c", "a"]), enc, np.float64)

        np.testing.assert_array_equal(codes, [2.0, 0.0])

    def test_offending_values_are_listed_in_order_of_appearance(self) -> None:
        """Deterministic, unlike scikit-learn's set-iteration order."""
        enc = _fit(list("abc"), np.float64, strict=True)

        with pytest.raises(ValueError) as excinfo:
            encode_categories(_object(["z", "y", "z"]), enc, np.float64)

        assert "['z', 'y']" in str(excinfo.value)


class TestMappingMemoisation:
    def test_mapping_is_reused(self) -> None:
        enc = _fit(list("abc"), np.float64, strict=False)

        assert _code_mapping(enc) is _code_mapping(enc)

    def test_mapping_survives_pickling(self) -> None:
        enc = _fit(list("abc"), np.float64, strict=False)
        _code_mapping(enc)

        restored = pickle.loads(pickle.dumps(enc))  # noqa: S301
        probe = _object(list("abc"))

        np.testing.assert_array_equal(
            encode_categories(probe, restored, np.float64),
            enc.transform(probe.reshape(-1, 1)).ravel(),  # type: ignore[no-untyped-call]
        )
