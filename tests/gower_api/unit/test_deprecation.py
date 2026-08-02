# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

"""Unit tests for the ``retired_args`` decorator."""

import warnings
from typing import Any

import pytest

from gower_metric.utils.deprecation import retired_args


@retired_args("is retired", "n_jobs", "backend")
def _compute(value: int, *, scale: int = 1) -> tuple[Any, ...]:
    return value, scale


class TestSilentCases:
    def test_call_without_retired_args_is_silent(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _compute(1) == (1, 1)

        assert caught == []

    def test_live_keyword_is_silent(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _compute(1, scale=3) == (1, 3)

        assert caught == []


class TestRetiredKeywordsAreAbsorbed:
    def test_retired_keyword_warns_and_is_discarded(self) -> None:
        with pytest.warns(DeprecationWarning, match="n_jobs is retired"):
            result = _compute(1, n_jobs=4)  # type: ignore[call-arg]

        assert result == (1, 1)

    def test_default_looking_value_still_warns(self) -> None:
        """The parameter no longer exists, so any value is equally meaningless."""
        with pytest.warns(DeprecationWarning, match="n_jobs"):
            _compute(1, n_jobs=-1)  # type: ignore[call-arg]

    def test_multiple_reported_alphabetically_in_one_warning(self) -> None:
        with pytest.warns(DeprecationWarning, match="backend, n_jobs") as record:
            _compute(1, n_jobs=4, backend="threading")  # type: ignore[call-arg]

        assert len(record) == 1
        assert str(record[0].message) == "backend, n_jobs is retired"

    def test_retired_and_live_keywords_mix(self) -> None:
        with pytest.warns(DeprecationWarning, match="backend"):
            assert _compute(1, backend="threading", scale=5) == (  # type: ignore[call-arg]
                1,
                5,
            )

    def test_unknown_keyword_still_raises(self) -> None:
        """Absorbing retired names must not turn into swallowing typos."""
        with pytest.raises(TypeError, match="unexpected keyword"):
            _compute(1, nonsense=1)  # type: ignore[call-arg]


class TestStalePositionalCalls:
    def test_extra_positional_raises_naming_the_retired_params(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            _compute(1, 4)  # type: ignore[misc]

        message = str(excinfo.value)
        assert "n_jobs, backend" in message
        assert "keyword" in message

    def test_allowed_positional_still_works(self) -> None:
        assert _compute(1) == (1, 1)


class TestMisuseIsRejected:
    def test_still_declared_parameter_raises_at_decoration_time(self) -> None:
        """Python binds a declared parameter before the wrapper can absorb it."""
        with pytest.raises(ValueError, match="still declares n_jobs"):

            @retired_args("is retired", "n_jobs")
            def _bad(value: int, n_jobs: int = -1) -> int:  # noqa: ARG001
                return value

    def test_no_names_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one parameter name"):
            retired_args("is retired")


class TestBehaviourPreserved:
    def test_metadata_is_preserved(self) -> None:
        assert _compute.__name__ == "_compute"

    def test_var_positional_functions_are_left_alone(self) -> None:
        @retired_args("is retired", "n_jobs")
        def _variadic(*args: int) -> tuple[Any, ...]:
            return args

        assert _variadic(1, 2, 3) == (1, 2, 3)
