from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.preprocessing import OrdinalEncoder

from gower_metric.core.config import Config
from gower_metric.core.exceptions import IllegalStateError
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
from gower_metric.utils.matrix.calculate_matrix import get_full_matrix
from gower_metric.utils.ranges import get_numeric_ranges
from gower_metric.utils.to_array import to_array
from gower_metric.utils.transformation import (
    validate_if_double_transformed,
    validate_if_transformed,
)
from gower_metric.weights.weights import get_weights


class Gower:
    """Compute Gower distance for mixed data types."""

    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize Gower with passed Config object.

        Args:
            config (Config): Configuration object containing all parameters needed for initialization.

        Raises:
            ValueError: If feature_types is not a non-empty dict.

        Example:
            >>> import pandas as pd
            >>> from gower_metric import Gower
            >>> from gower_metric.core.config import Config
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
            >>> feature_weights = {
            ...     0: 1.0,
            ...     1: 2.0,
            ...     2: 1.0,
            >>> cfg = Config(
            ...     feature_types=feature_types,
            ...     feature_weights=feature_weights,
            ... )
            >>> gower = Gower(cfg)

        """
        self.feature_types = config.feature_types

        self.feature_weights = config.feature_weights

        self.data_type: type[np.integer | np.floating] = (
            config.data_type if config.data_type is not None else np.float32
        )

        self.numeric_indices: list[int] = []
        self.categorical_nominal_indices: list[int] = []
        self.categorical_ordinal_indices: list[int] = []
        self.binary_asymmetric_indices: list[int] = []
        self.binary_symmetric_indices: list[int] = []
        self.ratio_scale_indices: list[int] = []
        self.ratio_ranges: np.ndarray = np.array([])
        self.numeric_ranges: np.ndarray = np.array([])

        self.scale_method: str = config.scale_method

        self.missing_strategy: str = config.missing_strategy
        self.categorical_ordinal_values_order = config.categorical_ordinal_values_order

        self.categorical_ordinal_calculation_type = (
            config.categorical_ordinal_calculation_type
        )

        self.scale_window: str | None = config.scale_window
        self.scale_window_type: str | None = config.scale_window_type

        self.k_neighbours = config.k_neighbours

        self.conditional_distances = config.conditional_distances

        self.conditional_distances_threshold_coeff = (
            config.conditional_distances_threshold_coeff
        )

        self._is_fitted: bool = False
        self._is_transformed: bool = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "Gower":
        """Fit the Gower model by computing numeric feature ranges.

        Args:
            X (np.ndarray | pd.DataFrame): shape of (n_samples, n_features).
                For DataFrame inputs, column names in feature_types are converted to indices.

        Returns:
            Gower: The fitted instance.

        Raises:
            ValueError: For incorrect input data and configuration parameters.
            ValueError: If input data is already transformed.

        Example:
            >>> import pandas as pd
            >>> from gower_metric import Gower
            >>> from gower_metric.core.config import Config
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
            >>> feature_weights = {
            ...     0: 1.0,
            ...     1: 2.0,
            ...     2: 1.0,
            >>> cfg = Config(
            ...     feature_types=feature_types,
            ...     feature_weights=feature_weights,
            ... )
            >>> gower = Gower(cfg).fit(data)

        """
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)

            ft: dict[int, str] = {}
            for k, t in self.feature_types.items():
                if isinstance(k, str):
                    if k not in cols:
                        msg = f"Column name '{k}' not found in DataFrame."
                        raise ValueError(msg)
                    ft[cols.index(k)] = t
                else:
                    ft[k] = t
            self.feature_types = ft  # type: ignore[assignment]

            if self.categorical_ordinal_values_order:
                for k in list(self.categorical_ordinal_values_order.keys()):
                    if isinstance(k, str):
                        if k not in cols:
                            msg = f"Column name '{k}' specified for categorical ordinal values not found in DataFrame."
                            raise ValueError(msg)
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

        self.n_feats = arr.shape[1]
        if self.conditional_distances:
            self.p_cat = (
                len(self.binary_symmetric_indices)
                + len(self.binary_asymmetric_indices)
                + len(self.categorical_nominal_indices)
                + len(self.categorical_ordinal_indices)
            )

        if self.ratio_scale_indices:
            self.ratio_ranges = get_numeric_ranges(
                arr,
                self.ratio_scale_indices,
                self.scale_method,
            )
        else:
            self.ratio_ranges = np.array([])

        if self.numeric_indices:
            self.numeric_ranges = get_numeric_ranges(
                arr,
                self.numeric_indices,
                self.scale_method,
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

        self.cat_ord_metadata: dict[int | str, dict[str, Any]] = {}
        for j in self.categorical_ordinal_indices:
            col = arr[:, j]
            if self.categorical_ordinal_values_order is None:
                msg = "Categorical ordinal values order is missing"
                raise ValueError(msg)
            ranks_map, mn, mx = map_ordered_values(
                self.categorical_ordinal_values_order[j],
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

        self.weights = get_weights(
            n_features=self.n_feats,
            config=self.feature_weights,
        )

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the input DataFrame or ndarray to contain only floats.

        Updates the Gower model feature ranges calculated by the fitting.

        Useful when applying 'gower' distance metrics in scikit-learn methods
        (e.g., KNN) requiring numeric input exclusively.

        Args:
            X (np.ndarray | pd.DataFrame): shape of (n_samples, n_features).
                For DataFrame inputs, column names in feature_types are converted to indices.

        Returns:
            X_new: Transformed input data.

        Raises:
            IllegalStateError: If fit(X) was not performed before calling transform(X).
            ValueError: For incorrect input data and configuration parameters.

        Example:
            >>> import pandas as pd
            >>> from gower_metric import Gower
            >>> from gower_metric.core.config import Config
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
            >>> cfg = Config(
            ...     feature_types=feature_types,
            ...     feature_weights=feature_weights,
            ... )
            >>> gower = Gower(cfg).fit(data)
            >>> data_transformed = gower.transform(data)

        """
        if not self._is_fitted:
            msg = "Operation not allowed: model is not fitted"
            raise IllegalStateError(msg)

        validate_if_double_transformed(self._is_transformed)

        is_df = isinstance(X, pd.DataFrame)
        if isinstance(X, pd.DataFrame):
            df: pd.DataFrame = X
            X_arr = df.to_numpy()
        else:
            arr: np.ndarray = X
            X_arr = np.asarray(arr, dtype=object)

        transformed_columns: list[np.ndarray] = []

        for col_idx, ftype in self.feature_types.items():
            col = df.iloc[:, col_idx].to_numpy() if is_df else X_arr[:, col_idx]

            if ftype in ("binary_asymmetric", "binary_symmetric"):
                transformed_col = np.zeros(col.shape[0], dtype=float)
                for i, v in enumerate(col):
                    if pd.isna(v):
                        transformed_col[i] = np.nan
                    elif str(v).lower() in ("true", "1", "yes", "1.0"):
                        transformed_col[i] = 1.0
                    else:
                        transformed_col[i] = 0.0

            elif ftype == "categorical_ordinal":
                if self.categorical_ordinal_values_order is None:
                    msg = "Categorical ordinal values order is missing"
                    raise ValueError(msg)
                enc = OrdinalEncoder(
                    categories=[
                        self.categorical_ordinal_values_order[col_idx],
                    ],
                    dtype=self.data_type,
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                )
                transformed_col = (
                    enc.fit_transform(np.array(col).reshape(-1, 1))
                    .astype(self.data_type)
                    .ravel()
                )

            elif ftype == "categorical_nominal":
                enc = OrdinalEncoder(
                    dtype=self.data_type,
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                )
                enc.fit(np.array(col).reshape(-1, 1))
                transformed_col = (
                    enc.transform(np.array(col).reshape(-1, 1))
                    .astype(self.data_type)
                    .ravel()
                )
            else:
                transformed_col = col.astype(self.data_type)

            transformed_columns.append(transformed_col)

        transformed_data: np.ndarray = np.column_stack(transformed_columns)

        for col_idx, _ in (
            (i, t) for i, t in self.feature_types.items() if t == "categorical_ordinal"
        ):
            self.cat_ord_metadata[col_idx]["ranks"] = {
                v: v for v in self.cat_ord_metadata[col_idx]["ranks"].values()
            }

        self._is_transformed = True
        if is_df:
            df_transformed = pd.DataFrame(
                transformed_data,
                columns=df.columns,
                index=df.index,
                dtype=self.data_type,
            )
            df_transformed.attrs["transformed"] = True
            return df_transformed
        dtype = np.dtype(self.data_type, metadata={"transformed": True})

        return transformed_data.astype(dtype)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Fit to data, then transform it.

        Args:
            X (np.ndarray | pd.DataFrame): shape of (n_samples, n_features).
                For DataFrame inputs, column names in feature_types are converted to indices.

        Returns:
            X_new: Transformed input data.

        """
        self.fit(X)
        return self.transform(X)

    def __call__(self, a: Any, b: Any) -> np.floating | np.integer:
        """Compute the Gower distance between two records.

        Args:
            a (Any): First record of data.
            b (Any): Second record of data.

        Returns:
            np.floating | np.integer: Gower distance in [0,1], or np.nan if no features are comparable.

        Raises:
            IllegalStateError: If fit(X) was not called before computing distance.

        Example:            >>> import pandas as pd
            >>> from gower_metric import Gower
            >>> from gower_metric.core.config import Config
            >>> data = pd.DataFrame({
            ...     'feature1': [[1.0], [2.0], [3.0], [4.0]],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            >>> feature_types = {
            ...     'feature1': 'numeric_interval',
            ...     'feature2': 'categorical_nominal',
            ... }
            >>> cfg = Config(
            ...     feature_types=feature_types,
            ... )
            >>> gower = Gower(cfg).fit(data)
            >>> distance = gower(data.iloc[0], data.iloc[1])

        """
        if not self._is_fitted:
            msg = "Must call .fit(X) before computing distances."
            raise IllegalStateError(msg)

        if self._is_transformed:
            validate_if_transformed(a)
            validate_if_transformed(b)

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

        cat_ord_sum, cat_ord_count = categorical_ordinal_component(
            Xn,
            Yn,
            self.categorical_ordinal_indices,
            metadata=self.cat_ord_metadata,
            missing_strategy=self.missing_strategy,
            calculation_type=self.categorical_ordinal_calculation_type,
            weights=cat_ord_w,
        )

        if self.conditional_distances:
            cat_sum = 0.0
            cat_cnt = 0.0

            if self.binary_asymmetric_indices:
                cat_sum += bin_asym_sum[0, 0]
                cat_cnt += bin_asym_count[0, 0]

            if self.binary_symmetric_indices:
                cat_sum += bin_sym_sum[0, 0]
                cat_cnt += bin_sym_count[0, 0]

            if self.categorical_nominal_indices:
                cat_sum += cat_nom_sum[0, 0]
                cat_cnt += cat_nom_count[0, 0]

            if self.categorical_ordinal_indices:
                cat_sum += cat_ord_sum[0, 0]
                cat_cnt += cat_ord_count[0, 0]

            if cat_cnt == 0:
                return self.data_type(np.nan)

            cat_dist = cat_sum / cat_cnt
            threshold = self.conditional_distances_threshold_coeff / self.p_cat

            if cat_dist > threshold:
                return self.data_type(1.0)

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

        if self.conditional_distances:
            total_sum = num_sum + ratio_sum
            total_count = num_count + ratio_count
        else:
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
            return self.data_type(np.nan)

        return self.data_type(total_sum[0, 0] / total_count[0, 0])

    def similarity(self, a: Any, b: Any) -> np.floating | np.integer:
        """Compute the Gower similarity between two records.

        Args:
            a (Any): First record of data.
            b (Any): Second record of data.

        Returns:
            float: Gower similarity in [0,1], defined as 1 - distance(a, b).

        Example:
            >>> from gower_metric import Gower
            >>> from gower_metric.core.config import Config
            >>> data = pd.DataFrame({
            ...     'feature1': [[1.0], [2.0], [3.0], [4.0]],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            >>> feature_types = {
            ...     'feature1': 'numeric_interval',
            ...     'feature2': 'categorical_nominal',
            ... }
            >>> cfg = Config(
            ...     feature_types=feature_types,
            ... )
            >>> gower = Gower(cfg).fit(data)
            >>> similarity = gower.similarity(data.iloc[0], data.iloc[1])

        """
        return 1.0 - self(a, b)

    def matrix(
        self,
        X: pd.DataFrame | np.ndarray,
        data_type: type[np.floating | np.integer] | None = None,
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
        """Return symmetric pairwise Gower distance matrix using joblib (parallel).

        Args:
            X (pd.DataFrame | np.ndarray): shape of (n_samples, n_features).
            data_type (type[np.floating | np.integer] | None): data type used for the output distance matrix.
                If None, uses the data_type from the Gower instance configuration.
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
                >>> from gower_metric import Gower
                >>> from gower_metric.core.config import Config
                >>> data = pd.DataFrame({
                ...     'feature1': [[1.0], [2.0], [3.0], [4.0]],
                ...     'feature2': ['A', 'B', 'A', 'C'],
                ...     'feature3': [0, 1, 0, 1],
                >>> feature_types = {
                ...     'feature1': 'numeric_interval',
                ...     'feature2': 'categorical_nominal',
                ...     'feature3': 'binary_symmetric',
                ... }
                >>> cfg = Config(
                ...     feature_types=feature_types,
                ... )
                >>> gower = Gower(cfg).fit(data)
                >>> similarity_matrix = gower.matrix(
                ...     data,
                ...     matrix_type='similarity',
                ...     convert_to_sparse=True,
                ...     sparse_type='csr'
                ... )

            Using similarity matrix and sparse output:
                >>> import pandas as pd
                >>> from gower_metric import Gower
                >>> from gower_metric.core.config import Config
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
                >>> cfg = Config(
                ...     feature_types=feature_types,
                ... )
                >>> gower = Gower(cfg).fit(data)
                >>> similarity_matrix = gower.matrix(
                ...     data,
                ...     matrix_type='similarity',
                ...     convert_to_sparse=True,
                ...     sparse_type='csr'
                ... )

        """
        if not self._is_fitted:
            self.fit(X)
            msg = "Calling .fit(X) inside .matrix(X)."
            raise Warning(msg)

        if self._is_transformed:
            validate_if_transformed(X)

        if data_type is None:
            data_type = self.data_type

        return get_full_matrix(
            self,
            X,
            data_type=data_type,
            n_jobs=n_jobs,
            verbose=verbose,
            matrix_type=matrix_type,
            convert_to_sparse=convert_to_sparse,
            sparse_type=sparse_type,
            backend=backend,
        )
