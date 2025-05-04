from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..utils.to_array import to_array
from ..utils.ranges import get_numeric_ranges
from ..utils.kde_types.silverman import silverman_bandwidth
from ..utils.validators import (
    validate_feature_types,
    validate_scale_method,
    validate_missing_strategy,
    validate_categorical_ordinal_calculation_type,
    validate_scale_window_and_type,
    validate_weights_type,
)
from ..utils.cat_ord_ut import get_ranks_mapping, get_cardinalities_mapping
from ..weights.weights import get_weights
from ..distances.numeric_interval import numeric_distance_matrix
from ..distances.categorical_nominal import nominal_distance_matrix
from ..distances.categorical_ordinal import ordinal_distance_matrix
from ..distances.binary_asymmetric import binary_asymmetric_distance_matrix
from ..distances.binary_symmetric import binary_symmetric_distance_matrix
from ..distances.ratio_scale_interval import ratio_scale_distance_matrix


class GowerSimilarity:
    """
    Copmute Gower similarity for mixed data types.    
    """

    def __init__(
        self,
        feature_types: Dict[Union[int, str], str],
        feature_weights: Optional[Union[Dict[Union[int, str], float], str]] = None,
        scale: Optional[str] = None,
        missing_strategy: Optional[str] = None,
        categorical_ordinal_calculation_type: Optional[str] = None,
        scale_window: Optional[str] = None,
        scale_window_type: Optional[str] = None,
    ) -> None:
        """
        Initialize GowerSimilarity with explicit feature type and weight mappings.

        Args:
            feature_types: Mapping of column indices (or DataFrame column names) to
                specific type.
            feature_weights: Optional mapping of column indices (or names) to a float weight.
                If None or "uniform", all features will have equal weight of 1. Otherwise,
                the weights must be a dictionary mapping feature indices to weights, i.e.
                {0: 1.0, 1: 2.0, ...}.
            scale: Optional scaling method for numeric features. Can be 'range' or 'iqr'.
                Default is 'range' if omitted.
            missing_strategy: Optional strategy for handling missing values. Can be 'ignore',
                'max_dist' or 'raise_error'. Default is 'ignore' if omitted.
            categorical_ordinal_calculation_type: Optional calculation type for categorical
                ordinal features. Can be 'kaufman' or 'podani'. Default is 'kaufman' if omitted.
            scale_window: Optional scaling window for numeric or ratio features. Can be None or 'kde'.
                Default is None if omitted.
            scale_window_type: Optional type of scaling window. Can be None or 'silverman'.
                Default is None if omitted, not recommended to use without scale_window.

        Raises:
            ValueError: If feature_types is not a non-empty dict.
        """
        validate_feature_types(feature_types)
        self.feature_types = feature_types

        self.feature_weights = feature_weights or {}
        validate_weights_type(self.feature_weights)

        self.numeric_indices = [
            i for i, t in feature_types.items() if t == "numeric"
        ]
        self.categorical_nominal_indices = [
            i for i, t in feature_types.items() if t == "categorical_nominal"
        ]
        self.categorical_ordinal_indices = [
            i for i, t in feature_types.items() if t == "categorical_ordinal"
        ]
        self.binary_asymmetric_indices = [
            i for i, t in feature_types.items() if t == "binary_asymmetric"
        ]
        self.binary_symmetric_indices = [
            i for i, t in feature_types.items() if t == "binary_symmetric"
        ]
        self.ratio_scale_indices = [
            i for i, t in feature_types.items() if t == "ratio_scale_interval"
        ]
        self.ratio_ranges: np.ndarray = np.array([])
        self.numeric_ranges: np.ndarray = np.array([])

        self.scale_method: Optional[str] = (scale or 'range').lower()
        validate_scale_method(self.scale_method)

        self.missing_strategy: Optional[str] = (missing_strategy
                                                or 'ignore').lower()
        validate_missing_strategy(self.missing_strategy)

        self.categorical_ordinal_calculation_type: Optional[str] = (
            categorical_ordinal_calculation_type or 'kaufman').lower()
        validate_categorical_ordinal_calculation_type(
            self.categorical_ordinal_calculation_type
        )

        self.scale_window: Optional[str] = scale_window if scale_window else None
        self.scale_window_type: Optional[str] = scale_window_type if scale_window_type else None
        validate_scale_window_and_type(self.scale_window, self.scale_window_type)

        self._is_fitted = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "GowerSimilarity":
        """
        Fit the GowerSimilarity model by computing numeric feature ranges.

        Args:
            X: pandas DataFrame or NumPy array of shape (n_samples, n_features).
            For DataFrame inputs, column names in feature_types are converted to indices.

        Returns:
            self: The fitted instance.
        """
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)

            ft: Dict[int, str] = {}
            for k, t in self.feature_types.items():
                if isinstance(k, str):
                    if k not in cols:
                        raise ValueError(
                            f"Column name '{k}' not found in DataFrame.")
                    ft[cols.index(k)] = t
                else:
                    ft[k] = t
            self.feature_types = ft

            self.numeric_indices = [i for i, t in ft.items() if t == "numeric"]
            self.categorical_nominal_indices = [
                i for i, t in ft.items() if t == "categorical_nominal"
            ]
            self.categorical_ordinal_indices = [
                i for i, t in ft.items() if t == "categorical_ordinal"
            ]
            self.binary_asymmetric_indices = [
                i for i, t in ft.items() if t == "binary_asymmetric"
            ]
            self.binary_symmetric_indices = [
                i for i, t in ft.items() if t == "binary_symmetric"
            ]
            self.ratio_scale_indices = [
                i for i, t in ft.items() if t == "ratio_scale_interval"
            ]
        arr = X.to_numpy(dtype=object) if isinstance(
            X, pd.DataFrame) else np.array(X, dtype=object)

        if self.ratio_scale_indices:
            self.ratio_ranges = get_numeric_ranges(arr,
                                                   self.ratio_scale_indices,
                                                   self.scale_method)
        else:
            self.ratio_ranges = np.array([])

        if self.numeric_indices:
            self.numeric_ranges = get_numeric_ranges(arr, self.numeric_indices,
                                                     self.scale_method)
        else:
            self.numeric_ranges = np.array([])

        if self.scale_window == "kde" and self.scale_window_type == "silverman":
            self._h_ratio = np.array([
                silverman_bandwidth(arr[:, j].astype(float))
                for j in self.ratio_scale_indices
            ], dtype=float)
            self._h_numeric = np.array([
                silverman_bandwidth(arr[:, j].astype(float))
                for j in self.numeric_indices
            ], dtype=float)
        else:
            self._h_ratio = None
            self._h_numeric = None

        self.cat_ord_metadata: Dict[int, Dict[str, Any]] = {}
        for j in self.categorical_ordinal_indices:
            col = arr[:, j]
            ranks_map, mn, mx = get_ranks_mapping(col)
            counts_map, _ = get_cardinalities_mapping(col)
            counts_arr = np.asarray([counts_map[v] for v in ranks_map.keys()], dtype=float)

            self.cat_ord_metadata[j] = {
                "ranks" : ranks_map,
                "denom" : (mx - mn) if mn is not None else 0,
                "counts": counts_arr,
                "min"   : mn,
                "max"   : mx,
            }

        n_feats = arr.shape[1]
        self.weights = get_weights(
            n_features=n_feats,
            config=self.feature_weights,
        )

        self._is_fitted = True
        return self

    def distance(self, a: Any, b: Any) -> float:
        """
        Compute the Gower distance between two records.

        Args:
            a: First record, can be numpy.ndarray, pandas.Series, or sequence of feature values.
            b: Second record, same restrictions as 'a'.

        Returns:
            float: Gower distance in [0,1], or np.nan if no features are comparable.

        Raises:
            ValueError: If fit(X) was not called before computing distance.
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

        num_sum, num_count = numeric_distance_matrix(
            Xn,
            Yn,
            self.numeric_indices,
            ranges=self.numeric_ranges,
            missing_strategy=self.missing_strategy,
            weights=num_w,
            scale_window=self.scale_window,
            h=self._h_numeric,
        )

        cat_nom_sum, cat_nom_count = nominal_distance_matrix(
            Xn,
            Yn,
            self.categorical_nominal_indices,
            missing_strategy=self.missing_strategy,
            weights=cat_nom_w)

        cat_ord_sum, cat_ord_count = ordinal_distance_matrix(
            Xn,
            Yn,
            self.categorical_ordinal_indices,
            metadata=self.cat_ord_metadata,
            missing_strategy=self.missing_strategy,
            calculation_type=self.categorical_ordinal_calculation_type,
            weights=cat_ord_w,
        )

        bin_asym_sum, bin_asym_count = binary_asymmetric_distance_matrix(
            Xn,
            Yn,
            self.binary_asymmetric_indices,
            missing_strategy=self.missing_strategy,
            weights=bin_asym_w,
        )

        bin_sym_sum, bin_sym_count = binary_symmetric_distance_matrix(
            Xn,
            Yn,
            self.binary_symmetric_indices,
            missing_strategy=self.missing_strategy,
            weights=bin_sym_w,
        )

        ratio_sum, ratio_count = ratio_scale_distance_matrix(
            Xn,
            Yn,
            self.ratio_scale_indices,
            ranges=self.ratio_ranges,
            missing_strategy=self.missing_strategy,
            weights=ratio_w,
            scale_window=self.scale_window,
            h=self._h_ratio,
        )
        total_sum = num_sum + cat_nom_sum + cat_ord_sum + bin_asym_sum + bin_sym_sum + ratio_sum
        total_count = num_count + cat_nom_count + cat_ord_count + bin_asym_count + bin_sym_count + ratio_count
        if total_count[0, 0] == 0:
            return float('nan')

        return float(total_sum[0, 0] / total_count[0, 0])

    def similarity(self, a: Any, b: Any) -> float:
        """
        Compute the Gower similarity between two records.

        Args:
            a: First record, same format as for distance().
            b: Second record, same format as for distance().

        Returns:
            float: Gower similarity in [0,1], defined as 1 - distance(a, b).
        """
        return 1.0 - self.distance(a, b)
    
    def __call__(self, *args, **kwds):
        # TODO: add numba support
        pass
