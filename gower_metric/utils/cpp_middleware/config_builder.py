# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING

import numpy as np

from gower_metric.cpp import (
    HAS_FLOAT16,
    CppConfig,
    CppConfigData,
    CppConfigDataF,
    CppConfigDataH,
    CppConfigF,
    CppConfigH,
)
from gower_metric.utils.ranges import _bounds_in_precision
from gower_metric.utils.transforms.encoding import _code_mapping, _own_codes

if TYPE_CHECKING:
    from gower_metric.core.metric import Gower


def fitted_codes(model: "Gower") -> dict[int, dict[object, float]]:
    """Give the code of each fitted value of every nominal and binary feature."""
    binary = {**model.binary_symmetric_metadata, **model.binary_asymmetric_metadata}
    return {
        **{j: _code_mapping(e) for j, e in model.nominal_metadata.items()},
        **{j: _own_codes(b["mapping"]) or b["mapping"] for j, b in binary.items()},
    }


_ConfigClasses = (
    tuple[type[CppConfigH], type[CppConfigDataH]]
    | tuple[type[CppConfigF], type[CppConfigDataF]]
    | tuple[type[CppConfig], type[CppConfigData]]
)
_CLASSES_BY_ITEMSIZE: dict[int, _ConfigClasses] = {
    np.dtype(np.float32).itemsize: (CppConfigF, CppConfigDataF),
    np.dtype(np.float64).itemsize: (CppConfig, CppConfigData),
}
if HAS_FLOAT16:
    _CLASSES_BY_ITEMSIZE[np.dtype(np.float16).itemsize] = (
        CppConfigH,
        CppConfigDataH,
    )

_NO_FLOAT16 = (
    "data_type=numpy.float16 needs a C++ compiler with _Float16, which this "
    "build of gower-metric does not have (MSVC, most likely). Use "
    "numpy.float32 or numpy.float64 instead. See the compatibility matrix in "
    "the README."
)


def build_cpp_config(self: "Gower") -> CppConfig | CppConfigF | CppConfigH:
    """Build the native config matching ``self.data_type``'s precision.

    Returns:
        CppConfig | CppConfigF | CppConfigH: The populated native config
        (float64 / float32 / float16 respectively).

    Raises:
        ValueError: If the feature types do not cover every column, or if
            ``data_type`` is a width the kernel lacks.
        NotImplementedError: If ``data_type`` is float16 on a build without
            ``_Float16``.

    """
    if any(i not in self.feature_types for i in range(self.n_feats)):
        msg = "Missmatched number of feature types"
        raise ValueError(msg)

    itemsize = np.dtype(self.data_type).itemsize
    if itemsize == np.dtype(np.float16).itemsize and not HAS_FLOAT16:
        raise NotImplementedError(_NO_FLOAT16)
    if itemsize not in _CLASSES_BY_ITEMSIZE:
        msg = (
            f"data_type=numpy.{self.data_type.__name__} is {itemsize * 8}-bit "
            "here, and the C++ kernel computes in float32 or float64 (float16 "
            "too where built with _Float16). Use numpy.float64; calculate_matrix "
            "can still return this type through its own data_type argument."
        )
        raise ValueError(msg)

    config_cls, data_cls = _CLASSES_BY_ITEMSIZE[itemsize]

    ranges = np.zeros(self.n_feats, dtype=self.data_type)
    ranges[self.numeric_indices] = self.numeric_ranges
    ranges[self.ratio_scale_indices] = self.ratio_ranges

    bandwidths = np.zeros(self.n_feats, dtype=self.data_type)
    if self._h_numeric.size:
        bandwidths[self.numeric_indices] = self._h_numeric
    if self._h_ratio.size:
        bandwidths[self.ratio_scale_indices] = self._h_ratio

    lower = np.full(self.n_feats, -np.inf)
    upper = np.full(self.n_feats, np.inf)
    for indices, mins, maxs in (
        (self.numeric_indices, self.numeric_mins, self.numeric_maxs),
        (self.ratio_scale_indices, self.ratio_mins, self.ratio_maxs),
    ):
        low, high = _bounds_in_precision(mins, maxs, self.data_type)
        lower[indices] = low
        upper[indices] = high

    data = data_cls()
    data.feature_types = [self.feature_types[i] for i in range(self.n_feats)]
    weights = np.array(self.weights, dtype=float)
    first = [*self.binary_symmetric_indices, *self.binary_asymmetric_indices]
    first += self.categorical_nominal_indices
    if self.conditional_distances and len(set(weights[first])) == 1:
        weights[first] = 1.0
    data.feature_weights = np.ascontiguousarray(weights, dtype=self.data_type)
    data.ranges = ranges
    data.bandwidths = bandwidths
    data.lower = lower.astype(self.data_type)
    data.upper = upper.astype(self.data_type)
    data.out_of_range = "clip" if self.skip_oor else self.out_of_range
    data.missing_strategy = self.missing_strategy
    data.categorical_ordinal_calculation_type = (
        self.categorical_ordinal_calculation_type
    )
    data.conditional_distances = self.conditional_distances
    data.conditional_distances_threshold_coeff = (
        self.conditional_distances_threshold_coeff
    )
    data.handle_unseen_binary_asymmetric = self.handle_unseen_binary_asymmetric
    data.handle_unseen_binary_symmetric = self.handle_unseen_binary_symmetric
    data.handle_unseen_categorical_nominal = self.handle_unseen_categorical_nominal
    data.handle_unseen_categorical_ordinal = self.handle_unseen_categorical_ordinal
    data.ordinal_counts = {
        int(j): self.cat_ord_metadata[j]["counts"].tolist()
        for j in self.categorical_ordinal_indices
    }
    data.excluded_features = [int(j) for j in self.excluded_features]
    data.category_values = {j: [*c.values()] for j, c in fitted_codes(self).items()}

    cfg = config_cls()
    cfg.configure_arguments(data)  # type: ignore[arg-type]
    return cfg
