# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

"""Compatibility tests for the ``calculate_matrix`` entry point."""

import warnings
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse as sp

from gower_metric import Config, Gower
from gower_metric.utils.matrix.distance import _temporary_skip_oor, calculate_matrix

FEATURE_TYPES: dict[int | str, str] = {
    0: "ratio_scale_interval",
    1: "categorical_nominal",
    2: "binary_symmetric",
}


def _data() -> npt.NDArray[np.object_]:
    return np.array(
        [
            [1.0, "a", 0],
            [2.0, "b", 1],
            [3.0, "a", 0],
            [4.0, "c", 1],
            [5.0, "b", 0],
        ],
        dtype=object,
    )


def _fitted() -> Gower:
    data = _data()
    return Gower(Config(feature_types=FEATURE_TYPES)).fit(data)


def _pairwise(
    gower: Gower,
    rows: npt.NDArray[np.object_],
    *,
    similarity: bool = False,
) -> npt.NDArray[np.floating]:
    call = gower.similarity if similarity else gower
    n = len(rows)
    return np.array(
        [[float(call(rows[i], rows[j])) for j in range(n)] for i in range(n)],
        dtype=gower.data_type,
    )


class TestMatchesScalarPath:
    @pytest.mark.parametrize("matrix_type", ["distance", "similarity"])
    def test_matches_pairwise_distances(self, matrix_type: str) -> None:
        gower = _fitted()
        data = _data()

        matrix = calculate_matrix(gower, data, matrix_type=matrix_type)
        expected = _pairwise(gower, data, similarity=matrix_type == "similarity")

        np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-6)  # type: ignore[arg-type]

    def test_data_type_override_is_applied(self) -> None:
        gower = _fitted()

        result = calculate_matrix(gower, _data(), data_type=np.float64)

        assert result.dtype == np.float64

    def test_sparse_output(self) -> None:
        gower = _fitted()
        data = _data()

        sparse = cast(
            "sp.coo_matrix",
            calculate_matrix(gower, data, convert_to_sparse=True, sparse_type="coo"),
        )

        assert sp.issparse(sparse)
        assert sparse.format == "coo"
        np.testing.assert_allclose(
            sparse.toarray(),
            _pairwise(gower, data),
            rtol=0,
            atol=1e-6,
        )

    def test_out_buffer_is_written_in_place(self) -> None:
        gower = _fitted()
        data = _data()
        buffer = np.empty((5, 5), dtype=gower.data_type)

        returned = calculate_matrix(gower, data, out=buffer)

        assert returned is buffer
        np.testing.assert_allclose(buffer, _pairwise(gower, data), rtol=0, atol=1e-6)

    def test_cross_matrix(self) -> None:
        gower = _fitted()
        data = _data()
        query = data[:2]

        cross = calculate_matrix(gower, data, Y=query)

        assert cross.shape == (5, 2)
        expected = np.array(
            [[float(gower(row, other)) for other in query] for row in data],
        )
        np.testing.assert_allclose(cross, expected, rtol=0, atol=1e-6)  # type: ignore[arg-type]


class TestValidation:
    def test_out_rejects_sparse_conversion(self) -> None:
        gower = _fitted()
        buffer = np.empty((5, 5), dtype=gower.data_type)

        with pytest.raises(ValueError, match=r"out cannot be combined"):
            calculate_matrix(gower, _data(), out=buffer, convert_to_sparse=True)

    def test_unknown_matrix_type_is_rejected(self) -> None:
        gower = _fitted()

        with pytest.raises(ValueError, match=r"Unknown matrix_type"):
            calculate_matrix(gower, _data(), matrix_type="nonsense")

    def test_sparse_input_is_rejected(self) -> None:
        gower = _fitted()

        with pytest.raises(ValueError, match=r"Sparse matrices"):
            calculate_matrix(gower, sp.csr_matrix(np.eye(3)))  # type: ignore[arg-type]

    def test_unfitted_model_is_fitted_with_warning(self) -> None:
        gower = Gower(Config(feature_types=FEATURE_TYPES))

        with pytest.warns(UserWarning, match=r"Calling \.fit\(X\)"):
            result = calculate_matrix(gower, _data())

        assert gower.is_fitted
        assert result.shape == (5, 5)


class TestRetiredJoblibParameters:
    def test_call_without_retired_params_is_silent(self) -> None:
        gower = _fitted()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calculate_matrix(gower, _data())

        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"n_jobs": 4}, "n_jobs"),
            ({"verbose": 1}, "verbose"),
            ({"backend": "threading"}, "backend"),
        ],
    )
    def test_non_default_warns(self, kwargs: dict[str, Any], expected: str) -> None:
        gower = _fitted()

        with pytest.warns(DeprecationWarning, match=expected):
            calculate_matrix(gower, _data(), **kwargs)

    def test_non_default_still_returns_correct_result(self) -> None:
        gower = _fitted()
        data = _data()

        with pytest.warns(DeprecationWarning, match="n_jobs"):
            result = calculate_matrix(gower, data, n_jobs=2)  # type: ignore[call-arg]

        np.testing.assert_allclose(result, _pairwise(gower, data), rtol=0, atol=1e-6)  # type: ignore[arg-type]

    def test_all_non_defaults_reported_together(self) -> None:
        gower = _fitted()

        with pytest.warns(
            DeprecationWarning,
            match="no longer has any effect",
        ) as record:
            calculate_matrix(  # type: ignore[call-arg]
                gower,
                _data(),
                n_jobs=4,
                verbose=1,
                backend="threading",
            )

        message = str(record[0].message)
        assert "backend" in message
        assert "n_jobs" in message
        assert "verbose" in message

    def test_stale_positional_call_is_rejected(self) -> None:
        """The old order was data_type/n_jobs/verbose/matrix_type.

        With the retired parameters gone, position 4 would now bind to
        ``matrix_type``. That must fail loudly rather than be reinterpreted.
        """
        gower = _fitted()

        with pytest.raises(TypeError) as excinfo:
            calculate_matrix(gower, _data(), None, 2, 0, "similarity")  # type: ignore[arg-type,misc]

        message = str(excinfo.value)
        assert "n_jobs, verbose, backend" in message
        assert "keyword" in message

    def test_retired_keywords_do_not_disturb_the_result(self) -> None:
        gower = _fitted()
        data = _data()

        with pytest.warns(DeprecationWarning, match="n_jobs"):
            result = calculate_matrix(  # type: ignore[call-arg]
                gower,
                data,
                n_jobs=2,
                matrix_type="similarity",
            )

        expected = _pairwise(gower, data, similarity=True)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)  # type: ignore[arg-type]


class TestTemporarySkipOor:
    def test_restores_previous_value(self) -> None:
        gower = _fitted()
        gower.skip_oor = False

        with _temporary_skip_oor(gower):
            assert gower.skip_oor is True

        assert gower.skip_oor is False

    def test_restores_after_exception(self) -> None:
        gower = _fitted()
        gower.skip_oor = False

        with pytest.raises(RuntimeError), _temporary_skip_oor(gower):
            raise RuntimeError

        assert gower.skip_oor is False


class TestOutBufferValidation:
    """A mismatched buffer must fail loudly, never be silently reallocated."""

    def test_non_array_is_rejected(self) -> None:
        gower = _fitted()

        with pytest.raises(TypeError, match=r"out must be a numpy ndarray"):
            calculate_matrix(gower, _data(), out=[[0.0] * 5] * 5)  # type: ignore[arg-type]

    def test_wrong_shape_is_rejected(self) -> None:
        gower = _fitted()
        buffer = np.empty((4, 5), dtype=gower.data_type)

        with pytest.raises(ValueError, match=r"out must have shape \(5, 5\)"):
            calculate_matrix(gower, _data(), out=buffer)

    def test_wrong_dtype_is_rejected(self) -> None:
        gower = _fitted()
        buffer = np.empty((5, 5), dtype=np.float64)

        with pytest.raises(ValueError, match=r"out must have dtype"):
            calculate_matrix(gower, _data(), out=buffer)

    def test_non_contiguous_is_rejected(self) -> None:
        gower = _fitted()
        buffer = np.empty((5, 10), dtype=gower.data_type)[:, ::2]

        with pytest.raises(ValueError, match=r"out must be C-contiguous"):
            calculate_matrix(gower, _data(), out=buffer)

    def test_read_only_is_rejected(self) -> None:
        gower = _fitted()
        buffer = np.empty((5, 5), dtype=gower.data_type)
        buffer.flags.writeable = False

        with pytest.raises(ValueError, match=r"out must be writable"):
            calculate_matrix(gower, _data(), out=buffer)


class TestNativeConfigLifecycle:
    def test_missing_native_config_is_rebuilt(self) -> None:
        """Unpickling drops the handle, so the matrix path must restore it."""
        gower = _fitted()
        gower.cpp_config = None

        result = calculate_matrix(gower, _data())

        assert gower.cpp_config is not None
        assert result.shape == (5, 5)

    def test_skip_oor_bypasses_the_range_check(self) -> None:
        train = np.array([[1.0, "a", 0], [5.0, "b", 1]], dtype=object)
        probe = np.array([[99.0, "a", 0], [5.0, "b", 1]], dtype=object)
        cfg = Config(
            feature_types=FEATURE_TYPES,
            out_of_range="error",
            skip_out_of_range_validation=True,
        )
        gower = Gower(cfg).fit(train)

        assert calculate_matrix(gower, probe).shape == (2, 2)
