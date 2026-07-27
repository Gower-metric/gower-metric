import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from gower_metric.cpp import (
    CppConfig,
    CppConfigData,
    CppConfigDataF,
    CppConfigDataH,
    CppConfigF,
    CppConfigH,
)

if TYPE_CHECKING:
    from gower_metric.core.metric import Gower


def _value_orders(
    order: Mapping[int | str, list[Any]] | None,
) -> dict[int, list[str]]:
    """Carry a per-column value-order mapping into the native config."""
    return {int(col): [str(v) for v in values] for col, values in (order or {}).items()}


_ConfigClasses = (
    tuple[type[CppConfigH], type[CppConfigDataH]]
    | tuple[type[CppConfigF], type[CppConfigDataF]]
    | tuple[type[CppConfig], type[CppConfigData]]
)
_CLASSES_BY_ITEMSIZE: dict[int, _ConfigClasses] = {
    np.dtype(np.float16).itemsize: (CppConfigH, CppConfigDataH),
    np.dtype(np.float32).itemsize: (CppConfigF, CppConfigDataF),
    np.dtype(np.float64).itemsize: (CppConfig, CppConfigData),
}


def build_cpp_config(self: "Gower") -> CppConfig | CppConfigF | CppConfigH:
    """Build the native config matching ``self.data_type``'s precision.

    Returns:
        CppConfig | CppConfigF | CppConfigH: The populated native config
        (float64 / float32 / float16 respectively).

    Raises:
        ValueError: If the feature types do not cover every column.

    """
    if any(i not in self.feature_types for i in range(self.n_feats)):
        msg = "Missmatched number of feature types"
        raise ValueError(msg)

    config_cls, data_cls = _CLASSES_BY_ITEMSIZE.get(
        np.dtype(self.data_type).itemsize,
        (CppConfig, CppConfigData),
    )

    ranges = np.zeros(self.n_feats, dtype=self.data_type)
    ranges[self.numeric_indices] = self.numeric_ranges
    ranges[self.ratio_scale_indices] = self.ratio_ranges

    bandwidths = np.zeros(self.n_feats, dtype=self.data_type)
    if self._h_numeric.size:
        bandwidths[self.numeric_indices] = self._h_numeric
    if self._h_ratio.size:
        bandwidths[self.ratio_scale_indices] = self._h_ratio

    data = data_cls()
    data.feature_types = [self.feature_types[i] for i in range(self.n_feats)]
    data.feature_weights = np.ascontiguousarray(self.weights, dtype=self.data_type)
    data.ranges = ranges
    data.bandwidths = bandwidths
    data.missing_strategy = self.missing_strategy
    data.scale_method = self.scale_method
    data.categorical_ordinal_calculation_type = (
        self.categorical_ordinal_calculation_type
    )
    data.conditional_distances = self.conditional_distances
    data.conditional_distances_threshold_coeff = (
        self.conditional_distances_threshold_coeff
    )
    data.discretization = self.discretization or ""
    data.silverman_constant = float(self.silverman_constant)
    data.k_neighbors = self.k_neighbors
    data.handle_unseen_binary_asymmetric = self.handle_unseen_binary_asymmetric
    data.handle_unseen_binary_symmetric = self.handle_unseen_binary_symmetric
    data.handle_unseen_categorical_nominal = self.handle_unseen_categorical_nominal
    data.handle_unseen_categorical_ordinal = self.handle_unseen_categorical_ordinal
    data.out_of_range = self.out_of_range
    data.skip_out_of_range_validation = self.skip_oor
    data.categorical_ordinal_values_order = _value_orders(
        self.categorical_ordinal_values_order,
    )
    data.binary_asymmetric_value_order = _value_orders(
        self.binary_asymmetric_value_order,
    )
    data.binary_symmetric_value_order = _value_orders(
        self.binary_symmetric_value_order,
    )
    data.ordinal_counts = {
        int(j): self.cat_ord_metadata[j]["counts"].tolist()
        for j in self.categorical_ordinal_indices
    }

    if self.categorical_ordinal_calculation_type == "podani":
        for j in self.categorical_ordinal_indices:
            counts = np.asarray(
                self.cat_ord_metadata[j]["counts"],
                dtype=self.data_type,
            )
            if counts.size == 0:
                continue
            mid = (counts - 1.0) / 2.0
            podani_denom = (counts.size - 1) - mid[0] - mid[-1]
            if podani_denom <= 0:
                msg = (
                    f"Podani denominator <= 0 for ordinal feature at index {j}. "
                    "Falling back to Kaufman method."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)

    cfg = config_cls()
    cfg.configure_arguments(data)  # type: ignore[arg-type]
    return cfg
