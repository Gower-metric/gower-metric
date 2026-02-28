"""Tests for Gower metric edge cases — error branches, call-before-fit, conditional distances."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from gower_metric import Config, Gower
from gower_metric.core.exceptions import IllegalStateError


class TestFitErrors:
    def test_fit_with_missing_column_name_raises(self) -> None:
        df = pd.DataFrame({"age": [1, 2], "income": [100, 200]})
        cfg = Config(feature_types={"age": "numeric", "nonexistent": "numeric"})
        gower = Gower(cfg)
        with pytest.raises(ValueError, match=r"Column name .* not found"):
            gower.fit(df)

    def test_fit_with_sparse_matrix_raises(self) -> None:
        data = scipy.sparse.csr_matrix([[1.0, 0], [0, 1.0]])
        cfg = Config(feature_types={0: "numeric", 1: "numeric"})
        gower = Gower(cfg)
        with pytest.raises(
            ValueError,
            match="Sparse matrices are currently not supported",
        ):
            gower.fit(data)


class TestCallBeforeFit:
    def test_call_before_fit_raises(self) -> None:
        cfg = Config(feature_types={0: "numeric"})
        gower = Gower(cfg)
        with pytest.raises(IllegalStateError, match=r"Must call .fit"):
            gower(np.array([1.0]), np.array([2.0]))

    def test_transform_before_fit_raises(self) -> None:
        cfg = Config(feature_types={0: "numeric"})
        gower = Gower(cfg)
        with pytest.raises(IllegalStateError):
            gower.transform(np.array([[1.0]]))


class TestMatrixAutoFit:
    def test_matrix_without_fit_raises_warning(self) -> None:
        data = np.array([[1.0, 0], [2.0, 1], [3.0, 0]])
        cfg = Config(feature_types={0: "numeric", 1: "binary_symmetric"})
        gower = Gower(cfg)
        with pytest.raises(Warning, match=r"Calling .fit"):
            gower.matrix(data)

    def test_matrix_without_fit_raises_warning_pandas(self) -> None:
        data = pd.DataFrame({"val": [1.0, 2.0, 3.0], "flag": [0, 1, 0]})
        cfg = Config(feature_types={"val": "numeric", "flag": "binary_symmetric"})
        gower = Gower(cfg)
        with pytest.raises(Warning, match=r"Calling .fit"):
            gower.matrix(data)


class TestMatrixWithDataType:
    def test_matrix_with_explicit_data_type(self) -> None:
        data = np.array([[1.0, 0], [2.0, 1], [3.0, 0]])
        cfg = Config(feature_types={0: "numeric", 1: "binary_symmetric"})
        gower = Gower(cfg).fit(data)
        result = gower.matrix(data, data_type=np.float64)
        assert result.dtype == np.float64

    def test_matrix_with_n_jobs_1(self) -> None:
        """Run matrix with n_jobs=1 so coverage tracks __compute_row_upper."""
        data = np.array([[1.0, 0], [2.0, 1], [3.0, 0]])
        cfg = Config(feature_types={0: "numeric", 1: "binary_symmetric"})
        gower = Gower(cfg).fit(data)
        result = gower.matrix(data, n_jobs=1)
        assert result.shape == (3, 3)
        np.testing.assert_almost_equal(result[0, 0], 0.0)

    def test_matrix_similarity_n_jobs_1(self) -> None:
        """Similarity matrix via n_jobs=1 for coverage of similarity branch + diagonal."""
        data = np.array([[1.0], [5.0], [10.0]])
        cfg = Config(feature_types={0: "numeric"})
        gower = Gower(cfg).fit(data)
        mat = gower.matrix(data, n_jobs=1, matrix_type="similarity")
        np.testing.assert_array_almost_equal(np.diag(mat), [1.0, 1.0, 1.0])

    def test_matrix_with_n_jobs_1_pandas(self) -> None:
        """n_jobs=1 matrix with pandas input."""
        data = pd.DataFrame({"val": [1.0, 2.0, 3.0], "flag": [0, 1, 0]})
        cfg = Config(feature_types={"val": "numeric", "flag": "binary_symmetric"})
        gower = Gower(cfg).fit(data)
        result = gower.matrix(data, n_jobs=1)
        assert result.shape == (3, 3)

    def test_matrix_similarity_n_jobs_1_pandas(self) -> None:
        """Similarity via n_jobs=1 with pandas."""
        data = pd.DataFrame({"x": [1.0, 5.0, 10.0]})
        cfg = Config(feature_types={"x": "numeric"})
        gower = Gower(cfg).fit(data)
        mat = gower.matrix(data, n_jobs=1, matrix_type="similarity")
        np.testing.assert_array_almost_equal(np.diag(mat), [1.0, 1.0, 1.0])


class TestConditionalDistancesEdgeCases:
    def test_conditional_with_binary_symmetric_indices(self) -> None:
        data = np.array(
            [
                [1.0, 0, "A"],
                [2.0, 1, "B"],
                [3.0, 0, "A"],
            ],
            dtype=object,
        )
        cfg = Config(
            feature_types={
                0: "numeric",
                1: "binary_symmetric",
                2: "categorical_nominal",
            },
            conditional_distances=True,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data[0], data[1])
        assert not np.isnan(dist)

    def test_conditional_with_cat_nominal_indices(self) -> None:
        data = np.array(
            [
                [1.0, "A"],
                [2.0, "B"],
            ],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "numeric", 1: "categorical_nominal"},
            conditional_distances=True,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data[0], data[1])
        assert not np.isnan(dist)

    def test_conditional_with_cat_ordinal_indices(self) -> None:
        data = np.array(
            [
                [1.0, "low"],
                [2.0, "high"],
            ],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "numeric", 1: "categorical_ordinal"},
            categorical_ordinal_values_order={1: ["low", "medium", "high"]},
            conditional_distances=True,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data[0], data[1])
        assert not np.isnan(dist)

    def test_conditional_all_categorical_nan_returns_nan(self) -> None:
        """When all categorical features are NaN, cat_cnt==0 → return NaN."""
        data = np.array(
            [
                [1.0, "A"],
                [2.0, "B"],
            ],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "numeric", 1: "categorical_nominal"},
            conditional_distances=True,
            missing_strategy="ignore",
        )
        gower = Gower(cfg).fit(data)
        a = np.array([1.0, np.nan], dtype=object)
        b = np.array([2.0, np.nan], dtype=object)
        dist = gower(a, b)
        assert np.isnan(dist)

    def test_conditional_with_binary_asymmetric(self) -> None:
        data = np.array(
            [
                [1.0, 0],
                [2.0, 1],
                [3.0, 0],
            ],
            dtype=object,
        )
        cfg = Config(
            feature_types={0: "numeric", 1: "binary_asymmetric"},
            conditional_distances=True,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data[0], data[1])
        assert not np.isnan(dist)

    def test_conditional_with_cat_nominal_pandas(self) -> None:
        """Conditional distances with pandas input."""
        data = pd.DataFrame({"val": [1.0, 2.0], "cat": ["A", "B"]})
        cfg = Config(
            feature_types={"val": "numeric", "cat": "categorical_nominal"},
            conditional_distances=True,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data.iloc[0], data.iloc[1])
        assert not np.isnan(dist)

    def test_conditional_with_binary_pandas(self) -> None:
        """Binary conditional distances with pandas."""
        data = pd.DataFrame({"val": [1.0, 2.0, 3.0], "flag": [0, 1, 0]})
        cfg = Config(
            feature_types={"val": "numeric", "flag": "binary_symmetric"},
            conditional_distances=True,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data.iloc[0], data.iloc[1])
        assert not np.isnan(dist)

    def test_similarity_method(self) -> None:
        data = np.array([[1.0], [2.0], [3.0]])
        cfg = Config(feature_types={0: "numeric"})
        gower = Gower(cfg).fit(data)
        sim = gower.similarity(data[0], data[1])
        dist = gower(data[0], data[1])
        assert np.isclose(float(sim), 1.0 - float(dist))

    def test_similarity_method_pandas(self) -> None:
        """Similarity with pandas Series input."""
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        cfg = Config(feature_types={"x": "numeric"})
        gower = Gower(cfg).fit(data)
        sim = gower.similarity(data.iloc[0], data.iloc[1])
        dist = gower(data.iloc[0], data.iloc[1])
        assert np.isclose(float(sim), 1.0 - float(dist))


class TestMatrixSimilarity:
    def test_similarity_matrix_diagonal_is_one(self) -> None:
        data = np.array([[1.0], [2.0], [3.0]])
        cfg = Config(feature_types={0: "numeric"})
        gower = Gower(cfg).fit(data)
        mat = gower.matrix(data, matrix_type="similarity")
        np.testing.assert_array_almost_equal(np.diag(mat), [1.0, 1.0, 1.0])

    def test_similarity_matrix_diagonal_pandas(self) -> None:
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        cfg = Config(feature_types={"x": "numeric"})
        gower = Gower(cfg).fit(data)
        mat = gower.matrix(data, matrix_type="similarity")
        np.testing.assert_array_almost_equal(np.diag(mat), [1.0, 1.0, 1.0])


class TestDtypeEdgeCases:
    def test_call_float64(self) -> None:
        """__call__ with float64 data type."""
        data = np.array([[1.0, "A"], [2.0, "B"]], dtype=object)
        cfg = Config(
            feature_types={0: "numeric", 1: "categorical_nominal"},
            data_type=np.float64,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data[0], data[1])
        assert 0.0 <= float(dist) <= 1.0

    def test_call_float64_pandas(self) -> None:
        data = pd.DataFrame({"val": [1.0, 2.0], "cat": ["A", "B"]})
        cfg = Config(
            feature_types={"val": "numeric", "cat": "categorical_nominal"},
            data_type=np.float64,
        )
        gower = Gower(cfg).fit(data)
        dist = gower(data.iloc[0], data.iloc[1])
        assert 0.0 <= float(dist) <= 1.0

    def test_matrix_float64_n_jobs_1(self) -> None:
        """n_jobs=1 matrix with float64."""
        data = np.array([[1.0], [5.0], [10.0]])
        cfg = Config(feature_types={0: "numeric"}, data_type=np.float64)
        gower = Gower(cfg).fit(data)
        mat = gower.matrix(data, n_jobs=1)
        assert mat.dtype == np.float64


class TestTransformDoesNotMutateFitMetadata:
    """Regression tests: transform() must not corrupt metadata used by __call__()."""

    def test_call_after_transform_ordinal(self) -> None:
        """Distance via __call__ must be identical before and after transform."""
        data = np.array([["low", 1.0], ["medium", 2.0], ["high", 3.0]], dtype=object)
        cfg = Config(
            feature_types={0: "categorical_ordinal", 1: "numeric"},
            categorical_ordinal_values_order={0: ["low", "medium", "high"]},
        )
        gower = Gower(cfg).fit(data)

        dist_before = gower(data[0], data[2])
        _ = gower.transform(data)
        dist_after = gower(data[0], data[2])

        assert not np.isnan(dist_after), "transform() corrupted ordinal metadata"
        assert pytest.approx(float(dist_before), rel=1e-12) == float(dist_after)

    def test_call_after_transform_ordinal_pandas(self) -> None:
        data = pd.DataFrame({"grade": ["low", "med", "high"], "val": [1.0, 2.0, 3.0]})
        cfg = Config(
            feature_types={"grade": "categorical_ordinal", "val": "numeric"},
            categorical_ordinal_values_order={"grade": ["low", "med", "high"]},
        )
        gower = Gower(cfg).fit(data)

        dist_before = gower(data.iloc[0], data.iloc[2])
        _ = gower.transform(data)
        dist_after = gower(data.iloc[0], data.iloc[2])

        assert not np.isnan(dist_after)
        assert pytest.approx(float(dist_before), rel=1e-12) == float(dist_after)

    def test_multiple_transforms_idempotent(self) -> None:
        """Calling transform() N times must not accumulate corruption."""
        data = np.array([["a", 1.0], ["b", 2.0], ["c", 3.0]], dtype=object)
        cfg = Config(
            feature_types={0: "categorical_ordinal", 1: "numeric"},
            categorical_ordinal_values_order={0: ["a", "b", "c"]},
        )
        gower = Gower(cfg).fit(data)

        dist_baseline = gower(data[0], data[2])
        for _ in range(5):
            _ = gower.transform(data)
            dist = gower(data[0], data[2])
            assert pytest.approx(float(dist_baseline), rel=1e-12) == float(dist)
