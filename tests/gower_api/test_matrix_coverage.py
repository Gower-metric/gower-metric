"""Tests for matrix calculation and sparse conversion edge cases."""

import numpy as np
import pandas as pd
import pytest

from gower_metric import Config, Gower
from gower_metric.utils.matrix.convert_matrix import get_scipy_sparse_matrix


class TestSparseMatrixConversion:
    def test_invalid_format_raises(self) -> None:
        data = np.eye(3, dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported matrix format"):
            get_scipy_sparse_matrix(data, matrix_format="bad_format")


class TestMatrixCompute:
    def _get_fitted_gower(self) -> tuple["Gower", np.ndarray]:
        data = np.array([[1.0], [5.0], [10.0]])
        cfg = Config(feature_types={0: "numeric"})
        return Gower(cfg).fit(data), data

    def _get_fitted_gower_pandas(self) -> tuple["Gower", pd.DataFrame]:
        data = pd.DataFrame({"value": [1.0, 5.0, 10.0]})
        cfg = Config(feature_types={"value": "numeric"})
        return Gower(cfg).fit(data), data

    def test_similarity_matrix_diagonal_ones(self) -> None:
        gower, data = self._get_fitted_gower()
        mat = gower.matrix(data, matrix_type="similarity")
        np.testing.assert_array_almost_equal(np.diag(mat), [1.0, 1.0, 1.0])
        assert mat[0, 1] == mat[1, 0]  # symmetric

    def test_distance_matrix_diagonal_zeros(self) -> None:
        gower, data = self._get_fitted_gower()
        mat = gower.matrix(data, matrix_type="distance")
        np.testing.assert_array_almost_equal(np.diag(mat), [0.0, 0.0, 0.0])

    def test_verbose_matrix(self) -> None:
        gower, data = self._get_fitted_gower()
        mat = gower.matrix(data, verbose=1)
        assert mat.shape == (3, 3)

    def test_sparse_similarity_matrix(self) -> None:
        """Similarity + convert_to_sparse covers branch 164→167."""
        gower, data = self._get_fitted_gower()
        gower.matrix(data, matrix_type="similarity", convert_to_sparse=True, sparse_type="csr")

    def test_similarity_matrix_pandas(self) -> None:
        """Matrix computation with pandas DataFrame."""
        gower, data = self._get_fitted_gower_pandas()
        mat = gower.matrix(data, matrix_type="similarity")
        np.testing.assert_array_almost_equal(np.diag(mat), [1.0, 1.0, 1.0])

    def test_distance_matrix_pandas(self) -> None:
        gower, data = self._get_fitted_gower_pandas()
        mat = gower.matrix(data, matrix_type="distance")
        np.testing.assert_array_almost_equal(np.diag(mat), [0.0, 0.0, 0.0])

    def test_matrix_float64_dtype(self) -> None:
        """Matrix with float64 output dtype."""
        data = np.array([[1.0], [5.0], [10.0]])
        cfg = Config(feature_types={0: "numeric"}, data_type=np.float64)
        gower = Gower(cfg).fit(data)
        mat = gower.matrix(data)
        assert mat.dtype == np.float64
        assert mat.shape == (3, 3)

    def test_matrix_float64_pandas(self) -> None:
        """Matrix with float64 + pandas input."""
        data = pd.DataFrame({"x": [1.0, 5.0, 10.0]})
        cfg = Config(feature_types={"x": "numeric"}, data_type=np.float64)
        gower = Gower(cfg).fit(data)
        mat = gower.matrix(data)
        assert mat.dtype == np.float64
