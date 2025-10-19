from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse

from gower_metric.distances.binary_asymmetric import (
    binary_asymmetric_component,
)
from gower_metric.distances.binary_symmetric import binary_symmetric_component
from gower_metric.distances.categorical_nominal import categorical_nominal_component
from gower_metric.distances.categorical_ordinal import categorical_ordinal_component
from gower_metric.distances.numeric_interval import numeric_component
from gower_metric.distances.ratio_scale_interval import ratio_scale_component
from gower_metric.utils.cat_ord_ut import (
    get_cardinalities_mapping,
    map_ordered_values,
)
from gower_metric.utils.kde_types.silverman import silverman_bandwidth
from gower_metric.utils.knn_bandwidth import knn_bandwidth
from gower_metric.utils.matrix.calculate_matrix import get_results_from_joblib
from gower_metric.utils.matrix.convert_matrix import get_scipy_sparse_matrix
from gower_metric.utils.ranges import get_numeric_ranges
from gower_metric.utils.to_array import to_array
from gower_metric.utils.validators import (
    validate_categorical_ordinal_calculation_type,
    validate_categorical_ordinal_values_order,
    validate_conditional_distances,
    validate_feature_types,
    validate_k_neighbours,
    validate_missing_strategy,
    validate_scale_method,
    validate_scale_window_and_type,
    validate_weights_type,
)
from gower_metric.weights.weights import get_weights


class Gower:
    """
    Compute Gower distance for mixed data types.
    """

    def __init__(
        self,
        feature_types: dict[int | str, str],
        feature_weights: dict[int, float] | str | None = None,
        scale: str = "range",
        missing_strategy: str = "ignore",
        categorical_ordinal_values_order: dict[int | str, list[str]] | None = None,
        categorical_ordinal_calculation_type: str = "kaufman",
        scale_window: str | None = None,
        scale_window_type: str | None = None,
        k_neighbours: int | None = None,
        conditional_distances: bool = False,
    ) -> None:
        """
        Initialize Gower with explicit feature type and weight mappings.

        Args:
            feature_types (dict[int | str, str]): Mapping of column indices (or DataFrame column names) to
                specific type.
            feature_weights (dict[int, float] | str | None): Optional mapping of column indices (or names) to a float weight.
                If None or "uniform", all features will have equal weight of 1. Otherwise,
                the weights must be a dictionary mapping feature indices to weights, i.e.
                {0: 1.0, 1: 2.0}.
            scale (str): Optional scaling method for numeric features. Can be 'range' or 'iqr'.
                Default is 'range' if omitted.
            missing_strategy (str): Optional strategy for handling missing values. Can be 'ignore',
                'max_dist' or 'raise_error'. Default is 'ignore' if omitted.
            categorical_ordinal_values_order (dict[int | str, list[str]] | None): Optional dict defining the order of the values contained in
                the columns of type 'categorical_ordinal'. Must contain values for all such columns.
            categorical_ordinal_calculation_type (str): Optional calculation type for categorical
                ordinal features. Can be 'kaufman' or 'podani'. Default is 'kaufman' if omitted.
            scale_window (Optional[str]): Optional scaling window for numeric or ratio features. Can be None, 'kde'
                or 'kNN'. Default is None if omitted.
            scale_window_type (Optional[str]): Optional type of scaling window. Can be None or 'silverman'.
                Default is None if omitted, not recommended to use without scale_window.
            k_neighbours (Optional[int]): Optional number of nearest neighbors for 'kNN' scaling window.
                Default is None if omitted. If k_neighbours is None or less than 1, it will be
                set to the square root of the number of points.
            conditional_distances (bool): Default to False. If set to True, two-step approach will be
                triggered to calculate formula. More information in references -> chapter 3.

        Raises:
            ValueError: If feature_types is not a non-empty dict.

        Example:
            >>> import pandas as pd
            >>> from gower_metric import Gower
            >>> data = pd.DataFrame({
            ...     'feature1': [[1.0], [2.0], [3.0], [4.0]],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            ...     'feature3': [0, 1, 0, 1],
            ... })
            >>> feature_types = {
            ...     'feature1': 'numeric_interval',
            ...     'feature2': 'categorical_nominal',
            ...     'feature3': 'binary_symmetric',
            ... }
            ... feature_weights = {
            ...     0: 1.0,
            ...     1: 2.0,
            ...     2: 1.0,
            >>> gower = Gower(feature_types=feature_types, feature_weights=feature_weights)
        """
        validate_feature_types(feature_types)
        self.feature_types: dict[int | str, str] = feature_types

        self.feature_weights = feature_weights or {}
        validate_weights_type(self.feature_weights)

        self.numeric_indices: list[int] = []
        self.categorical_nominal_indices: list[int] = []
        self.categorical_ordinal_indices: list[int] = []
        self.binary_asymmetric_indices: list[int] = []
        self.binary_symmetric_indices: list[int] = []
        self.ratio_scale_indices: list[int] = []
        self.ratio_ranges: np.ndarray = np.array([])
        self.numeric_ranges: np.ndarray = np.array([])

        self.scale_method: str = (scale or "range").lower()
        validate_scale_method(self.scale_method)

        self.missing_strategy: str = (missing_strategy or "ignore").lower()
        validate_missing_strategy(self.missing_strategy)

        if categorical_ordinal_values_order is None:
            categorical_ordinal_values_order = {}
        self.categorical_ordinal_values_order = categorical_ordinal_values_order
        validate_categorical_ordinal_values_order(
            self.categorical_ordinal_values_order, self.feature_types
        )

        self.categorical_ordinal_calculation_type: str = (
            categorical_ordinal_calculation_type or "kaufman"
        ).lower()
        validate_categorical_ordinal_calculation_type(
            self.categorical_ordinal_calculation_type
        )

        self.scale_window: str | None = scale_window if scale_window else None
        self.scale_window_type: str | None = (
            scale_window_type if scale_window_type else None
        )
        validate_scale_window_and_type(self.scale_window, self.scale_window_type)

        self.k_neighbours = k_neighbours if k_neighbours else None
        validate_k_neighbours(self.k_neighbours)

        self.conditional_distances = conditional_distances
        validate_conditional_distances(self.conditional_distances)

        self._is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "Gower":
        """
        Fit the Gower model by computing numeric feature ranges.

        Args:
            X (np.ndarray | pd.DataFrame): shape of (n_samples, n_features).
                For DataFrame inputs, column names in feature_types are converted to indices.

        Returns:
            Gower: The fitted instance.

        Example:
            >>> import pandas as pd
            >>> from gower_metric import Gower
            >>> data = pd.DataFrame({
            ...     'feature1': [[1.0], [2.0], [3.0], [4.0]],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            ...     'feature3': [0, 1, 0, 1],
            ... })
            >>> feature_types = {
            ...     'feature1': 'numeric_interval',
            ...     'feature2': 'categorical_nominal',
            ...     'feature3': 'binary_symmetric',
            ... }
            >>> gower = Gower(feature_types=feature_types).fit(data)
        """
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)

            ft: dict[int, str] = {}
            for k, t in self.feature_types.items():
                if isinstance(k, str):
                    if k not in cols:
                        raise ValueError(f"Column name '{k}' not found in DataFrame.")
                    ft[cols.index(k)] = t
                else:
                    ft[k] = t
            self.feature_types = ft  # type: ignore[assignment]

            if self.categorical_ordinal_values_order:
                for k in list(self.categorical_ordinal_values_order.keys()):
                    if isinstance(k, str):
                        if k not in cols:
                            raise ValueError(
                                f"Column name '{k}' specified for categorical ordinal values not found in DataFrame."
                            )
                        self.categorical_ordinal_values_order[cols.index(k)] = (
                            self.categorical_ordinal_values_order.pop(k)
                        )

        self.numeric_indices = [
            i
            for i, t in self.feature_types.items()
            if isinstance(i, int) and t == "numeric"
        ]
        self.categorical_nominal_indices = [
            i
            for i, t in self.feature_types.items()
            if isinstance(i, int) and t == "categorical_nominal"
        ]
        self.categorical_ordinal_indices = [
            i
            for i, t in self.feature_types.items()
            if isinstance(i, int) and t == "categorical_ordinal"
        ]
        self.binary_asymmetric_indices = [
            i
            for i, t in self.feature_types.items()
            if isinstance(i, int) and t == "binary_asymmetric"
        ]
        self.binary_symmetric_indices = [
            i
            for i, t in self.feature_types.items()
            if isinstance(i, int) and t == "binary_symmetric"
        ]
        self.ratio_scale_indices = [
            i
            for i, t in self.feature_types.items()
            if isinstance(i, int) and t == "ratio_scale_interval"
        ]
        arr = (
            X.to_numpy(dtype=object)
            if isinstance(X, pd.DataFrame)
            else np.array(X, dtype=object)
        )

        if self.ratio_scale_indices:
            self.ratio_ranges = get_numeric_ranges(
                arr, self.ratio_scale_indices, self.scale_method
            )
        else:
            self.ratio_ranges = np.array([])

        if self.numeric_indices:
            self.numeric_ranges = get_numeric_ranges(
                arr, self.numeric_indices, self.scale_method
            )
        else:
            self.numeric_ranges = np.array([])

        if self.scale_window == "kde" and self.scale_window_type == "silverman":
            self._h_ratio = np.array(
                [
                    silverman_bandwidth(arr[:, j].astype(float))
                    for j in self.ratio_scale_indices
                ],
                dtype=float,
            )
            self._h_numeric = np.array(
                [
                    silverman_bandwidth(arr[:, j].astype(float))
                    for j in self.numeric_indices
                ],
                dtype=float,
            )
        elif self.scale_window == "kNN":
            self._h_ratio = np.array(
                [
                    knn_bandwidth(arr[:, j].astype(float), k=self.k_neighbours)
                    for j in self.ratio_scale_indices
                ],
                dtype=float,
            )
            self._h_numeric = np.array(
                [
                    knn_bandwidth(arr[:, j].astype(float), k=self.k_neighbours)
                    for j in self.numeric_indices
                ],
                dtype=float,
            )
        else:
            self._h_ratio = np.empty(0)
            self._h_numeric = np.empty(0)

        self.cat_ord_metadata: dict[int, dict[str, Any]] = {}
        for j in self.categorical_ordinal_indices:
            col = arr[:, j]
            ranks_map, mn, mx = map_ordered_values(
                self.categorical_ordinal_values_order[j]
            )
            counts_map, _ = get_cardinalities_mapping(col)
            counts_arr = np.asarray([counts_map[v] for v in ranks_map], dtype=float)

            self.cat_ord_metadata[j] = {
                "ranks": ranks_map,
                "denom": (mx - mn) if mn is not None and mx is not None else 0,
                "counts": counts_arr,
                "min": mn,
                "max": mx,
            }

        n_feats = arr.shape[1]
        self.weights = get_weights(
            n_features=n_feats,
            config=self.feature_weights,
        )

        self._is_fitted = True

        self.p_cat = (
            len(self.binary_symmetric_indices)
            + len(self.binary_asymmetric_indices)
            + len(self.categorical_nominal_indices)
        )
        return self

    def __call__(self, a: Any, b: Any) -> float:
        """
        Compute the Gower distance between two records.

        Args:
            a (Any): First record of data.
            b (Any): Second record of data.

        Returns:
            float: Gower distance in [0,1], or np.nan if no features are comparable.

        Raises:
            ValueError: If fit(X) was not called before computing distance.

        Example:
            >>> import pandas as pd
            >>> from gower_metric import Gower
            >>> data = pd.DataFrame({
            ...     'feature1': [[1.0], [2.0], [3.0], [4.0]],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            ... })
            >>> gower = Gower(feature_types={0: 'numeric_interval', 1: 'categorical_nominal'}).fit(data)
            >>> distance = gower(data.iloc[0], data.iloc[1])
        """
        if not self._is_fitted:
            raise ValueError("Must call .fit(X) before computing distances.")

        x = to_array(a)
        y = to_array(b)
        Xn = x.reshape(1, -1)
        Yn = y.reshape(1, -1)

        num_w = self.weights[self.numeric_indices]
        cat_nom_w = self.weights[self.categorical_nominal_indices]
        cat_ord_w = self.weights[self.categorical_ordinal_indices]
        bin_asym_w = self.weights[self.binary_asymmetric_indices]
        bin_sym_w = self.weights[self.binary_symmetric_indices]
        ratio_w = self.weights[self.ratio_scale_indices]

        if not self.conditional_distances:
            num_sum, num_count = numeric_component(
                Xn,
                Yn,
                self.numeric_indices,
                ranges=self.numeric_ranges,
                h=self._h_numeric,
                missing_strategy=self.missing_strategy,
                weights=num_w,
                scale_window=self.scale_window,
            )

            cat_nom_sum, cat_nom_count = categorical_nominal_component(
                Xn,
                Yn,
                self.categorical_nominal_indices,
                missing_strategy=self.missing_strategy,
                weights=cat_nom_w,
            )

            cat_ord_sum, cat_ord_count = categorical_ordinal_component(
                Xn,
                Yn,
                self.categorical_ordinal_indices,
                metadata=self.cat_ord_metadata,
                missing_strategy=self.missing_strategy,
                calculation_type=self.categorical_ordinal_calculation_type,
                weights=cat_ord_w,
            )

            bin_asym_sum, bin_asym_count = binary_asymmetric_component(
                Xn,
                Yn,
                self.binary_asymmetric_indices,
                missing_strategy=self.missing_strategy,
                weights=bin_asym_w,
            )

            bin_sym_sum, bin_sym_count = binary_symmetric_component(
                Xn,
                Yn,
                self.binary_symmetric_indices,
                missing_strategy=self.missing_strategy,
                weights=bin_sym_w,
            )

            ratio_sum, ratio_count = ratio_scale_component(
                Xn,
                Yn,
                self.ratio_scale_indices,
                ranges=self.ratio_ranges,
                h=self._h_ratio,
                missing_strategy=self.missing_strategy,
                weights=ratio_w,
                scale_window=self.scale_window,
            )

            total_sum = (
                num_sum
                + cat_nom_sum
                + cat_ord_sum
                + bin_asym_sum
                + bin_sym_sum
                + ratio_sum
            )
            total_count = (
                num_count
                + cat_nom_count
                + cat_ord_count
                + bin_asym_count
                + bin_sym_count
                + ratio_count
            )
            if total_count[0, 0] == 0:
                return float("nan")

            return float(total_sum[0, 0] / total_count[0, 0])
        else:
            bin_asym_sum, bin_asym_count = binary_asymmetric_component(
                Xn,
                Yn,
                self.binary_asymmetric_indices,
                missing_strategy=self.missing_strategy,
                weights=bin_asym_w,
            )

            bin_sym_sum, bin_sym_count = binary_symmetric_component(
                Xn,
                Yn,
                self.binary_symmetric_indices,
                missing_strategy=self.missing_strategy,
                weights=bin_sym_w,
            )

            cat_nom_sum, cat_nom_count = categorical_nominal_component(
                Xn,
                Yn,
                self.categorical_nominal_indices,
                missing_strategy=self.missing_strategy,
                weights=cat_nom_w,
            )
            cat_sum = 0.0
            cat_cnt = 0.0

            if self.binary_asymmetric_indices:
                s, c = binary_asymmetric_component(
                    Xn,
                    Yn,
                    self.binary_asymmetric_indices,
                    missing_strategy=self.missing_strategy,
                    weights=bin_asym_w,
                )
                cat_sum += s[0, 0]
                cat_cnt += c[0, 0]

            if self.binary_symmetric_indices:
                s, c = binary_symmetric_component(
                    Xn,
                    Yn,
                    self.binary_symmetric_indices,
                    missing_strategy=self.missing_strategy,
                    weights=bin_sym_w,
                )
                cat_sum += s[0, 0]
                cat_cnt += c[0, 0]

            if self.categorical_nominal_indices:
                s, c = categorical_nominal_component(
                    Xn,
                    Yn,
                    self.categorical_nominal_indices,
                    missing_strategy=self.missing_strategy,
                    weights=cat_nom_w,
                )
                cat_sum += s[0, 0]
                cat_cnt += c[0, 0]

            if cat_cnt == 0:
                return float("nan")

            cat_dist = cat_sum / cat_cnt
            threshold = 1.0 / self.p_cat

            if cat_dist > threshold:
                return 1.0

            num_sum, num_count = numeric_component(
                Xn,
                Yn,
                self.numeric_indices,
                ranges=self.numeric_ranges,
                h=self._h_numeric,
                missing_strategy=self.missing_strategy,
                weights=num_w,
                scale_window=self.scale_window,
            )
            cat_ord_sum, cat_ord_count = categorical_ordinal_component(
                Xn,
                Yn,
                self.categorical_ordinal_indices,
                metadata=self.cat_ord_metadata,
                missing_strategy=self.missing_strategy,
                calculation_type=self.categorical_ordinal_calculation_type,
                weights=cat_ord_w,
            )
            ratio_sum, ratio_count = ratio_scale_component(
                Xn,
                Yn,
                self.ratio_scale_indices,
                ranges=self.ratio_ranges,
                h=self._h_ratio,
                missing_strategy=self.missing_strategy,
                weights=ratio_w,
                scale_window=self.scale_window,
            )

            total_sum = num_sum + cat_ord_sum + ratio_sum
            total_cnt = num_count + cat_ord_count + ratio_count

            if total_cnt[0, 0] == 0:
                return float("nan")
            return float(total_sum[0, 0] / total_cnt[0, 0])

    def similarity(self, a: Any, b: Any) -> float:
        """
        Compute the Gower similarity between two records.

        Args:
            a (Any): First record of data.
            b (Any): Second record of data.

        Returns:
            float: Gower similarity in [0,1], defined as 1 - distance(a, b).

        Example:
            >>> import pandas as pd
            >>> from gower_metric import Gower
            >>> data = pd.DataFrame({
            ...     'feature1': [[1.0], [2.0], [3.0], [4.0]],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            ... })
            >>> gower = Gower(feature_types={0: 'numeric_interval', 1: 'categorical_nominal'}).fit(data)
            >>> similarity = gower.similarity(data.iloc[0], data.iloc[1])
        """
        return 1.0 - self(a, b)

    def matrix(
        self,
        X: pd.DataFrame | np.ndarray,
        data_type: type[np.floating | np.integer] = np.float32,
        n_jobs: int = -1,
        verbose: int = 0,
        matrix_type: str = "distance",
        convert_to_sparse: bool = False,
        sparse_type: str = "csr",
        backend: str = "loky",
    ) -> (
        np.ndarray
        | scipy.sparse.csr_matrix
        | scipy.sparse.csc_matrix
        | scipy.sparse.coo_matrix
    ):
        """Compute symmetric pairwise Gower distance matrix using joblib (parallel).

        Args:
            X (pd.DataFrame | np.ndarray): shape of (n_samples, n_features).
            data_type (type[np.floating | np.integer]): data type for the output distance matrix, default np.float32.
            n_jobs (int): number of parallel jobs to run, -1 means using all processors. Default is -1.
            verbose (int): whether to show tqdm progress bar. Default is 0 (no progress bar).
            matrix_type (str): Type of matrix to compute, either 'distance' or 'similarity'.
                Default is 'distance'.
            convert_to_sparse (bool): Whether to convert the output dense matrix to a sparse format.
                Default is False.
            sparse_type (str): Type of sparse matrix to convert to, either 'csr', 'csc' or 'coo'.
                Default is 'csr'.
            backend (str): Backend to use for joblib parallelization. Default is 'loky'.

        Returns:
            np.ndarray | scipy.sparse.csr_matrix | scipy.sparse.csc_matrix | scipy.sparse.coo_matrix:
                Pairwise Gower distance or similarity matrix of shape (n_samples, n_samples) or sparse matrix.

        Raises:
            Warning: If fit(X) was not called before computing the matrix. In this case,
                the model will be fitted automatically.

        Examples:
            Basic usage:
                >>> import pandas as pd
                >>> from gower_metric import Gower
                >>> data = pd.DataFrame({
                ...     'feature1': [[1.0], [2.0], [3.0], [4.0]],
                ...     'feature2': ['A', 'B', 'A', 'C'],
                ...     'feature3': [0, 1, 0, 1],
                ... })
                >>> feature_types = {
                ...     'feature1': 'numeric_interval',
                ...     'feature2': 'categorical_nominal',
                ...     'feature3': 'binary_symmetric',
                ... }
                >>> gower = Gower(feature_types=feature_types).fit(data)
                >>> distance_matrix = gower.matrix(data)

            Using similarity matrix and sparse output:
                >>> import pandas as pd
                >>> from gower_metric import Gower
                >>> data = pd.DataFrame({
                ...     'feature1': [[1.0], [2.0], [3.0], [4.0]],
                ...     'feature2': ['A', 'B', 'A', 'C'],
                ...     'feature3': [0, 1, 0, 1],
                ... })
                >>> feature_types = {
                ...     'feature1': 'numeric_interval',
                ...     'feature2': 'categorical_nominal',
                ...     'feature3': 'binary_symmetric',
                ... }
                >>> gower = Gower(feature_types=feature_types).fit(data)
                >>> similarity_matrix = gower.matrix(
                ...     data,
                ...     matrix_type='similarity',
                ...     convert_to_sparse=True,
                ...     sparse_type='csr'
                ... )

        """
        if not self._is_fitted:
            self.fit(X)
            raise Warning("Calling .fit(X) inside .matrix(X).")

        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=object)
        else:
            arr = np.array(X, dtype=object)

        n: int = arr.shape[0]

        MATRIX: np.ndarray = np.zeros((n, n), dtype=data_type)

        results: list[tuple[int, np.ndarray]] = get_results_from_joblib(
            n_jobs=n_jobs,
            n=n,
            arr=arr,
            model=self,
            data_type=data_type,
            verbose=verbose,
            backend=backend,
            matrix_type=matrix_type,
        )

        for i, row in results:
            MATRIX[i] = row

        MATRIX += MATRIX.T

        if matrix_type == "distance":
            np.fill_diagonal(MATRIX, 0.0)
        elif matrix_type == "similarity":
            np.fill_diagonal(MATRIX, 1.0)

        if convert_to_sparse:
            MATRIX = get_scipy_sparse_matrix(
                MATRIX, matrix_format=sparse_type, data_type=data_type
            )

        return MATRIX
