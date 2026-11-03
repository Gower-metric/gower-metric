# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

"""Compatibility tests for the ``calculate_matrix`` entry point."""

import threading
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse as sp

from gower_metric import Config, Gower
from gower_metric._typing import MatrixType
from gower_metric.utils.matrix.distance import calculate_matrix, temporary_skip_oor

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
    def test_matches_pairwise_distances(self, matrix_type: MatrixType) -> None:
        gower = _fitted()
        data = _data()

        matrix = calculate_matrix(gower, data, matrix_type=matrix_type)
        expected = _pairwise(gower, data, similarity=matrix_type == "similarity")

        np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-6)

    def test_data_type_override_is_applied(self) -> None:
        gower = _fitted()

        result = calculate_matrix(gower, _data(), data_type=np.float64)

        assert result.dtype == np.float64

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
        np.testing.assert_allclose(cross, expected, rtol=0, atol=1e-6)


class TestValidation:
    def test_unknown_matrix_type_is_rejected(self) -> None:
        gower = _fitted()

        with pytest.raises(ValueError, match=r"Unknown matrix_type"):
            calculate_matrix(gower, _data(), matrix_type="nonsense")  # type: ignore[arg-type]

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


class TestTemporarySkipOor:
    def test_skips_the_check_without_touching_the_model(self) -> None:
        gower = _fitted()
        out_of_range = np.array([[99.0, "a", 0]], dtype=object)

        with temporary_skip_oor(gower):
            gower.transform(out_of_range)
            assert gower.skip_oor is False

        with pytest.raises(ValueError, match="Out-of-range"):
            gower.transform(out_of_range)

    def test_skips_only_in_its_own_thread(self) -> None:
        gower = _fitted()
        out_of_range = np.array([[99.0, "a", 0]], dtype=object)
        opened, checked = threading.Event(), threading.Event()
        raised: list[bool] = []

        def transform_while_open() -> None:
            opened.wait()
            try:
                gower.transform(out_of_range)
            except ValueError:
                raised.append(True)
            checked.set()

        other = threading.Thread(target=transform_while_open)
        other.start()
        with temporary_skip_oor(gower):
            opened.set()
            checked.wait()
        other.join()

        assert raised == [True]

    def test_restores_after_exception(self) -> None:
        gower = _fitted()
        gower.skip_oor = False

        with pytest.raises(RuntimeError), temporary_skip_oor(gower):
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
        """Only a width the kernel can write is accepted."""
        gower = _fitted()
        buffer = np.empty((5, 5), dtype=np.int64)

        with pytest.raises(ValueError, match=r"out must have dtype"):
            calculate_matrix(gower, _data(), out=cast("Any", buffer))

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


class TestRaiseErrorStrategyReachesPython:
    """``missing_strategy='raise_error'`` must raise, not abort the process."""

    N_ROWS = 400
    N_FEATURES = 20

    def _fitted_with_missing(self) -> tuple[Gower, npt.NDArray[np.floating]]:
        rng = np.random.default_rng(0)
        data = np.ascontiguousarray(
            rng.normal(size=(self.N_ROWS, self.N_FEATURES)),
            dtype=np.float32,
        )
        data[7, 3] = np.nan
        cfg = Config(
            feature_types=dict.fromkeys(range(self.N_FEATURES), "numeric"),
            missing_strategy="raise_error",
        )
        return Gower(cfg).fit(data), data

    def test_square_path_raises(self) -> None:
        gower, data = self._fitted_with_missing()

        with pytest.raises(ValueError, match=r"Missing values detected"):
            calculate_matrix(gower, data)

    def test_cross_path_raises(self) -> None:
        gower, data = self._fitted_with_missing()

        with pytest.raises(ValueError, match=r"Missing values detected"):
            calculate_matrix(gower, data, Y=data[:50])

    def test_clean_data_still_computes(self) -> None:
        gower, data = self._fitted_with_missing()
        clean = np.nan_to_num(data)

        result = calculate_matrix(gower, clean)

        assert result.shape == (self.N_ROWS, self.N_ROWS)
        assert np.isfinite(np.asarray(result)).all()
