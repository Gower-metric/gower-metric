# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import scipy.sparse

from gower_metric._typing import (
    AnyArray,
    BinaryMetadata,
    DataFrameOrArray,
    FloatArray,
    FloatDType,
    NativeKernel,
    OrdinalMetadata,
    Record,
)
from gower_metric.core.config import (
    Config,
    OutOfRangeStrategy,
    SkipOutOfRangeValidation,
)
from gower_metric.core.exceptions import IllegalStateError
from gower_metric.utils.binary_ut import (
    fit_binary_features,
)
from gower_metric.utils.cat_ord_ut import (
    get_cardinalities_mapping,
    map_ordered_values,
)
from gower_metric.utils.categorical_ut import (
    fit_nominal_features,
    fit_ordinal_features,
)
from gower_metric.utils.cpp_middleware.config_builder import build_cpp_config
from gower_metric.utils.discretization_types import (
    knn,
    silverman,
)
from gower_metric.utils.ranges import (
    enforce_oor_policy,
    get_numeric_bounds,
    get_numeric_ranges,
)
from gower_metric.utils.row_cache import RowEncodeCache
from gower_metric.utils.transforms import (
    transform_binary_asymmetric,
    transform_binary_symmetric,
    transform_categorical_nominal,
    transform_categorical_ordinal,
)
from gower_metric.weights.weights import get_weights

if TYPE_CHECKING:
    from sklearn.preprocessing import OrdinalEncoder

    from gower_metric.cpp import CppConfig, CppConfigF, CppConfigH

_NUMERIC_DTYPE_KINDS = "iufcm"
"""Dtype kind codes that ``np.issubdtype(dtype, np.number)`` accepts."""


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
            >>> from gower_metric import Config, Gower
            >>> data = pd.DataFrame({
            ...     'feature1': [1.0, 2.0, 3.0, 4.0],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            ...     'feature3': [0, 1, 0, 1],
            ... })
            >>> feature_types = {
            ...     'feature1': 'numeric',
            ...     'feature2': 'categorical_nominal',
            ...     'feature3': 'binary_symmetric',
            ... }
            >>> feature_weights = {
            ...     0: 1.0,
            ...     1: 2.0,
            ...     2: 1.0,
            ... }
            >>> cfg = Config(
            ...     feature_types=feature_types,
            ...     feature_weights=feature_weights,
            ... )
            >>> gower = Gower(cfg)

        """
        self.feature_types = config.feature_types

        self.feature_weights = config.feature_weights

        self.data_type: FloatDType = (
            config.data_type if config.data_type is not None else np.float32
        )

        self.numeric_indices: list[int] = []
        self.categorical_nominal_indices: list[int] = []
        self.categorical_ordinal_indices: list[int] = []
        self.binary_asymmetric_indices: list[int] = []
        self.binary_symmetric_indices: list[int] = []
        self.ratio_scale_indices: list[int] = []
        self.ratio_ranges: FloatArray = np.array([])
        self.numeric_ranges: FloatArray = np.array([])
        self.numeric_mins: FloatArray = np.array([])
        self.numeric_maxs: FloatArray = np.array([])
        self.ratio_mins: FloatArray = np.array([])
        self.ratio_maxs: FloatArray = np.array([])

        self.scale_method: str = config.scale_method

        self.missing_strategy: str = config.missing_strategy
        self.categorical_ordinal_values_order = config.categorical_ordinal_values_order

        self.categorical_ordinal_calculation_type = (
            config.categorical_ordinal_calculation_type
        )

        self.discretization: str | None = config.discretization
        self.silverman_constant: int | float = config.silverman_constant

        self.k_neighbors = config.k_neighbors

        self.conditional_distances = config.conditional_distances

        self.conditional_distances_threshold_coeff = (
            config.conditional_distances_threshold_coeff
        )

        self.handle_unseen_binary_asymmetric = config.handle_unseen_binary_asymmetric
        self.binary_asymmetric_value_order = config.binary_asymmetric_value_order

        self.handle_unseen_binary_symmetric = config.handle_unseen_binary_symmetric
        self.binary_symmetric_value_order = config.binary_symmetric_value_order

        self.handle_unseen_categorical_nominal = (
            config.handle_unseen_categorical_nominal
        )
        self.handle_unseen_categorical_ordinal = (
            config.handle_unseen_categorical_ordinal
        )

        self.out_of_range: OutOfRangeStrategy = config.out_of_range
        self.skip_oor: SkipOutOfRangeValidation = config.skip_out_of_range_validation

        self._is_fitted: bool = False
        self._row_cache = RowEncodeCache()
        self.cpp_config: CppConfig | CppConfigF | CppConfigH | None = None
        self.binary_symmetric_metadata: dict[int, BinaryMetadata] = {}
        self.binary_asymmetric_metadata: dict[int, BinaryMetadata] = {}
        self.nominal_metadata: dict[int, OrdinalEncoder] = {}
        self.ordinal_metadata: dict[int, OrdinalEncoder] = {}

    @property
    def is_fitted(self) -> bool:
        """Whether the Gower instance has been fitted.

        Returns
        -------
        bool: True if the instance has been fitted and is ready for use; otherwise False.

        """
        return getattr(self, "_is_fitted", False)

    def __getstate__(self) -> dict[str, object]:
        """Return picklable state, dropping the native config handle.

        Returns:
            dict[str, object]: Instance ``__dict__`` with ``cpp_config`` set to None.

        """
        state = self.__dict__.copy()
        state["cpp_config"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore instance state, re-seeding the fields a pickle may predate.

        Args:
            state (dict[str, object]): State produced by ``__getstate__``.

        """
        self.__dict__.update(state)
        self.__dict__.setdefault("cpp_config", None)
        self.__dict__.setdefault("_row_cache", RowEncodeCache())

    def fit(self, X: DataFrameOrArray) -> "Gower":  # noqa: PLR0912, PLR0915
        """Fit the Gower model by computing numeric feature ranges.

        Args:
            X (DataFrameOrArray): shape of (n_samples, n_features).
                For DataFrame inputs, column names in feature_types are converted to indices.
                Sparse matrices are not supported as direct input.

        Returns:
            Gower: The fitted instance.

        Raises:
            ValueError: For incorrect input data and configuration parameters.
            ValueError: If input data is already transformed.

        Example:
            >>> import pandas as pd
            >>> from gower_metric import Config, Gower
            >>> data = pd.DataFrame({
            ...     'feature1': [1.0, 2.0, 3.0, 4.0],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            ...     'feature3': [0, 1, 0, 1],
            ... })
            >>> feature_types = {
            ...     'feature1': 'numeric',
            ...     'feature2': 'categorical_nominal',
            ...     'feature3': 'binary_symmetric',
            ... }
            >>> feature_weights = {
            ...     0: 1.0,
            ...     1: 2.0,
            ...     2: 1.0,
            ... }
            >>> cfg = Config(
            ...     feature_types=feature_types,
            ...     feature_weights=feature_weights,
            ... )
            >>> gower = Gower(cfg).fit(data)

        """
        if scipy.sparse.issparse(X):
            msg = "Sparse matrices are currently not supported as direct input. Please convert to dense."
            raise ValueError(msg)

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

            for order_dict, order_name in [
                (self.categorical_ordinal_values_order, "categorical ordinal"),
                (self.binary_asymmetric_value_order, "binary asymmetric"),
                (self.binary_symmetric_value_order, "binary symmetric"),
            ]:
                if order_dict:
                    for k in list(order_dict.keys()):
                        if isinstance(k, str):
                            if k not in cols:  # pragma: no cover
                                msg = f"Column name '{k}' specified for {order_name} values not found in DataFrame."
                                raise ValueError(msg)
                            values = order_dict.pop(k)
                            order_dict[cols.index(k)] = values  # type: ignore[assignment]

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

        int_keys = [k for k in self.feature_types if isinstance(k, int)]
        if int_keys:
            max_idx = max(int_keys)
            if max_idx >= self.n_feats:
                msg = (
                    f"feature_types references column index {max_idx}, "
                    f"but data has only {self.n_feats} columns."
                )
                raise ValueError(msg)

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
            self.ratio_mins, self.ratio_maxs = get_numeric_bounds(
                arr,
                self.ratio_scale_indices,
            )
        else:
            self.ratio_ranges = np.array([])
            self.ratio_mins = np.array([])
            self.ratio_maxs = np.array([])

        if self.numeric_indices:
            self.numeric_ranges = get_numeric_ranges(
                arr,
                self.numeric_indices,
                self.scale_method,
            )
            self.numeric_mins, self.numeric_maxs = get_numeric_bounds(
                arr,
                self.numeric_indices,
            )
        else:
            self.numeric_ranges = np.array([])
            self.numeric_mins = np.array([])
            self.numeric_maxs = np.array([])

        if self.discretization == silverman.NAME:
            self._h_ratio = np.array(
                [
                    silverman.bandwidth(
                        arr[:, j].astype(float),
                        c=self.silverman_constant,
                    )
                    for j in self.ratio_scale_indices
                ],
                dtype=float,
            )
            self._h_numeric = np.array(
                [
                    silverman.bandwidth(
                        arr[:, j].astype(float),
                        c=self.silverman_constant,
                    )
                    for j in self.numeric_indices
                ],
                dtype=float,
            )
        elif self.discretization == knn.NAME:
            self._h_ratio = np.array(
                [
                    knn.bandwidth(arr[:, j].astype(float), k=self.k_neighbors)
                    for j in self.ratio_scale_indices
                ],
                dtype=float,
            )
            self._h_numeric = np.array(
                [
                    knn.bandwidth(arr[:, j].astype(float), k=self.k_neighbors)
                    for j in self.numeric_indices
                ],
                dtype=float,
            )
        else:
            self._h_ratio = np.empty(0)
            self._h_numeric = np.empty(0)

        self.cat_ord_metadata: dict[int | str, OrdinalMetadata] = {}
        for j in self.categorical_ordinal_indices:
            col = arr[:, j]
            if self.categorical_ordinal_values_order is None:
                msg = "Categorical ordinal values order is missing"
                raise ValueError(msg)
            ranks_map, mn, mx = map_ordered_values(
                self.categorical_ordinal_values_order[j],
                data_type=self.data_type,
            )
            counts_map, _ = get_cardinalities_mapping(col)
            counts_arr = np.asarray(
                [
                    counts_map.get(v, 0)
                    for v in self.categorical_ordinal_values_order[j]
                ],
                dtype=float,
            )

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

        self.binary_symmetric_metadata = fit_binary_features(
            arr,
            self.binary_symmetric_indices,
            binary_value_order=self.binary_symmetric_value_order,  # type: ignore[arg-type]
        )

        self.binary_asymmetric_metadata = fit_binary_features(
            arr,
            self.binary_asymmetric_indices,
            binary_value_order=self.binary_asymmetric_value_order,  # type: ignore[arg-type]
        )

        self.nominal_metadata = fit_nominal_features(
            arr,
            self.categorical_nominal_indices,
            self.data_type,
            handle_unseen=self.handle_unseen_categorical_nominal,
        )

        self.ordinal_metadata = fit_ordinal_features(
            arr,
            self.categorical_ordinal_indices,
            self.categorical_ordinal_values_order,
            self.data_type,
            handle_unseen=self.handle_unseen_categorical_ordinal,
        )

        self.cpp_config = build_cpp_config(self)
        self._row_cache.clear()

        self._is_fitted = True
        return self

    def transform(self, X: DataFrameOrArray) -> pd.DataFrame | FloatArray:
        """Transform the input DataFrame or ndarray to contain only floats.

        Useful when applying 'gower' distance metrics in scikit-learn methods
        (e.g., KNN) requiring numeric input exclusively.

        Note:
            - Sparse matrices are not supported.
            - Model ranges and parameters are NOT updated by this method (it is stateless).

        Args:
            X (DataFrameOrArray): shape of (n_samples, n_features).
                For DataFrame inputs, column names in feature_types are converted to indices.

        Returns:
            X_new: Transformed input data.

        Raises:
            IllegalStateError: If fit(X) was not performed before calling transform(X).
            ValueError: For incorrect input data and configuration parameters.

        Example:
            >>> import pandas as pd
            >>> from gower_metric import Config, Gower
            >>> data = pd.DataFrame({
            ...     'feature1': [1.0, 2.0, 3.0, 4.0],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            ...     'feature3': [0, 1, 0, 1],
            ... })
            >>> feature_types = {
            ...     'feature1': 'numeric',
            ...     'feature2': 'categorical_nominal',
            ...     'feature3': 'binary_symmetric',
            ... }
            >>> cfg = Config(
            ...     feature_types=feature_types,
            ... )
            >>> gower = Gower(cfg).fit(data)
            >>> data_transformed = gower.transform(data)

        """
        if not self._is_fitted:
            msg = "Operation not allowed: model is not fitted"
            raise IllegalStateError(msg)

        is_df = isinstance(X, pd.DataFrame)
        X_arr: AnyArray
        if isinstance(X, pd.DataFrame):
            df: pd.DataFrame = X
            X_arr = df.to_numpy()
        else:
            X_arr = np.asarray(X, dtype=object)

        if not self.skip_oor:
            enforce_oor_policy(
                X,
                strategy=self.out_of_range,
                numeric_indices=self.numeric_indices,
                numeric_mins=self.numeric_mins,
                numeric_maxs=self.numeric_maxs,
                ratio_scale_indices=self.ratio_scale_indices,
                ratio_mins=self.ratio_mins,
                ratio_maxs=self.ratio_maxs,
                stacklevel=2,
            )

        n_rows = df.shape[0] if is_df else X_arr.shape[0]
        transformed_data: FloatArray = np.empty(
            (n_rows, self.n_feats),
            dtype=self.data_type,
        )

        for col_idx_raw, ftype in sorted(self.feature_types.items()):
            col_idx = int(col_idx_raw)

            col: AnyArray
            col = df.iloc[:, col_idx].to_numpy() if is_df else X_arr[:, col_idx]

            if ftype == "binary_asymmetric":
                transformed_col = transform_binary_asymmetric(
                    col=col,
                    col_idx=col_idx,
                    metadata=self.binary_asymmetric_metadata[col_idx],
                    handle_unseen=self.handle_unseen_binary_asymmetric,
                )

            elif ftype == "binary_symmetric":
                transformed_col = transform_binary_symmetric(
                    col=col,
                    col_idx=col_idx,
                    metadata=self.binary_symmetric_metadata[col_idx],
                    handle_unseen=self.handle_unseen_binary_symmetric,
                )

            elif ftype == "categorical_ordinal":
                if col_idx not in self.ordinal_metadata:  # pragma: no cover
                    msg = f"Ordinal metadata missing for column {col_idx}"
                    raise ValueError(msg)

                transformed_col = transform_categorical_ordinal(
                    col=col,
                    col_idx=col_idx,
                    enc=self.ordinal_metadata[col_idx],
                    handle_unseen=self.handle_unseen_categorical_ordinal,
                    data_type=self.data_type,
                )

            elif ftype == "categorical_nominal":
                if col_idx not in self.nominal_metadata:  # pragma: no cover
                    msg = f"Nominal metadata missing for column {col_idx}"
                    raise ValueError(msg)

                transformed_col = transform_categorical_nominal(
                    col=col,
                    col_idx=col_idx,
                    enc=self.nominal_metadata[col_idx],
                    handle_unseen=self.handle_unseen_categorical_nominal,
                    data_type=self.data_type,
                )
            else:
                transformed_col = col.astype(self.data_type)

            transformed_data[:, col_idx] = transformed_col

        if is_df:
            return pd.DataFrame(
                transformed_data,
                columns=df.columns,
                index=df.index,
                copy=False,
            )

        return transformed_data

    def fit_transform(self, X: DataFrameOrArray) -> pd.DataFrame | FloatArray:
        """Fit to data, then transform it.

        Args:
            X (DataFrameOrArray): shape of (n_samples, n_features).
                For DataFrame inputs, column names in feature_types are converted to indices.

        Returns:
            X_new: Transformed input data.

        """
        self.fit(X)
        return self.transform(X)

    def __call__(self, a: Record, b: Record) -> np.floating:
        """Compute the Gower distance between two records.

        Args:
            a (Record): First record of data.
            b (Record): Second record of data.

        Returns:
            np.floating: Gower distance in [0,1], or np.nan if no features are comparable.

        Raises:
            IllegalStateError: If fit(X) was not called before computing distance.

        Example:
            >>> import pandas as pd
            >>> from gower_metric import Config, Gower
            >>> data = pd.DataFrame({
            ...     'feature1': [1.0, 2.0, 3.0, 4.0],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            ...     })
            >>> feature_types = {
            ...     'feature1': 'numeric',
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

        if self.cpp_config is None:
            self.cpp_config = build_cpp_config(self)

        a_arr = np.asarray(a)
        b_arr = np.asarray(b)

        x: FloatArray
        y: FloatArray
        if (
            a_arr.dtype.kind in _NUMERIC_DTYPE_KINDS
            and b_arr.dtype.kind in _NUMERIC_DTYPE_KINDS
        ):
            x = np.ascontiguousarray(a_arr, dtype=self.data_type).reshape(-1)
            y = np.ascontiguousarray(b_arr, dtype=self.data_type).reshape(-1)
        else:
            x, y = self._row_cache.encode_pair(a, b, self.transform, self.data_type)

        kernel = cast("NativeKernel", self.cpp_config)
        return self.data_type(kernel.calculate_distance(x, y))

    def similarity(self, a: Record, b: Record) -> np.floating:
        """Compute the Gower similarity between two records.

        Args:
            a (Record): First record of data.
            b (Record): Second record of data.

        Returns:
            np.floating: Gower similarity in [0,1], defined as 1 - distance(a, b).

        Example:
            >>> import pandas as pd
            >>> from gower_metric import Config, Gower
            >>> data = pd.DataFrame({
            ...     'feature1': [1.0, 2.0, 3.0, 4.0],
            ...     'feature2': ['A', 'B', 'A', 'C'],
            ... })
            >>> feature_types = {
            ...     'feature1': 'numeric',
            ...     'feature2': 'categorical_nominal',
            ... }
            >>> cfg = Config(
            ...     feature_types=feature_types,
            ... )
            >>> gower = Gower(cfg).fit(data)
            >>> similarity = gower.similarity(data.iloc[0], data.iloc[1])

        """
        return self.data_type(1.0 - self(a, b))
