from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..utils.to_array import to_array
from ..utils.ranges import get_numeric_ranges
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
        feature_weights: Optional[Dict[Union[int, str], float]] = None,
        scale: Optional[str] = None,
    ) -> None:
        """
        Initialize GowerSimilarity with explicit feature type and weight mappings.

        Args:
            feature_types: Mapping of column indices (or DataFrame column names) to
                        specific type.
            feature_weights: Optional mapping of column indices (or names) to a float weight
                            (default is 1.0 for all features if omitted).
            scale: Optional scaling method for numeric features. Can be 'range' or 'iqr'.
                Default is 'range' if omitted.

        Raises:
            ValueError: If feature_types is not a non-empty dict.
        """
        # TODO: add support for other type + auto check names
        if not isinstance(feature_types, dict) or not feature_types:
            raise ValueError(
                """feature_types must be a non-empty dict mapping columns to: 'numeric', 'categorical_nominal',
                'categorical_ordinal', 'binary_asymmetric', 'binary_symmetric', 'ratio_scale_interval'.
                """)
        self.feature_types = feature_types

        # TODO: add support for other weights selection
        self.feature_weights = feature_weights or {}

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
        self.scale_method: Optional[str] = scale.lower() if scale else 'range'
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

        num_w = (np.array(
            [self.feature_weights.get(i, 1.0) for i in self.numeric_indices],
            dtype=float) if self.numeric_indices else None)
        cat_nom_w = (np.array([
            self.feature_weights.get(i, 1.0)
            for i in self.categorical_nominal_indices
        ],
                              dtype=float)
                     if self.categorical_nominal_indices else None)
        cat_ord_w = (np.array([
            self.feature_weights.get(i, 1.0)
            for i in self.categorical_ordinal_indices
        ],
                              dtype=float)
                     if self.categorical_ordinal_indices else None)
        bin_asym_w = (np.array([
            self.feature_weights.get(i, 1.0)
            for i in self.binary_asymmetric_indices
        ],
                               dtype=float)
                      if self.binary_asymmetric_indices else None)
        bin_sym_w = (np.array([
            self.feature_weights.get(i, 1.0)
            for i in self.binary_symmetric_indices
        ],
                              dtype=float)
                     if self.binary_symmetric_indices else None)
        ratio_w = (np.array([
            self.feature_weights.get(i, 1.0) for i in self.ratio_scale_indices
        ],
                            dtype=float) if self.ratio_scale_indices else None)
        num_sum, num_count = numeric_distance_matrix(
            Xn,
            Yn,
            self.numeric_indices,
            ranges=self.numeric_ranges,
            weights=num_w,
        )

        cat_nom_sum, cat_nom_count = nominal_distance_matrix(
            Xn, Yn, self.categorical_nominal_indices, weights=cat_nom_w)

        cat_ord_sum, cat_ord_count = ordinal_distance_matrix(
            Xn,
            Yn,
            self.categorical_ordinal_indices,
            weights=cat_ord_w,
        )

        bin_asym_sum, bin_asym_count = binary_asymmetric_distance_matrix(
            Xn,
            Yn,
            self.binary_asymmetric_indices,
            weights=bin_asym_w,
        )

        bin_sym_sum, bin_sym_count = binary_symmetric_distance_matrix(
            Xn,
            Yn,
            self.binary_symmetric_indices,
            weights=bin_sym_w,
        )

        ratio_sum, ratio_count = ratio_scale_distance_matrix(
            Xn,
            Yn,
            self.ratio_scale_indices,
            ranges=self.ratio_ranges,
            weights=ratio_w,
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
