# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

import math
import operator
from collections.abc import Mapping, Sequence
from typing import Any, Literal, get_args

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.types import StrictBool

FeatureType = Literal[
    "numeric",
    "categorical_nominal",
    "categorical_ordinal",
    "binary_asymmetric",
    "binary_symmetric",
    "ratio_scale_interval",
]
UnseenStrategy = Literal["warning", "error", "missing"]
OutOfRangeStrategy = Literal["clip", "warning", "error"]
WeightsType = Literal["uniform"] | Mapping[int, float]
FeatureTypes = Mapping[Any, str]
OrdinalValueOrders = Mapping[Any, Sequence[str]]
BinaryValueOrders = Mapping[Any, Sequence[object]]


class Config(BaseModel, extra="forbid"):
    """Configuration object to initialize Gower.

    Args:
        feature_types (dict[int | str, str]): Mapping of column indices (or DataFrame column names) to
            specific type.
        feature_weights (dict[int, float] | str | None): Optional mapping of column indices to a float weight.
            If None or "uniform", all features will have equal weight of 1. Otherwise,
            the weights must be a dictionary mapping feature indices to weights, i.e.
            {0: 1.0, 1: 2.0}.
        data_type (type[np.floating] | None): Optional data type required for __call__, transformation and matrix method.
            If omitted, np.float32 will be used.
        scale_method (str): Optional scaling method for numeric features. Can be 'range' or 'iqr'.
            Default is 'range' if omitted.
        discretization (str | None): Optional type of adaptive discretization. Can be None, 'silverman' or 'knn'.
            Default is None if omitted. When enabled, it reduces the disproportionate influence of categorical variables on the final result.
        silverman_constant (float): Optional flag to determine the value of parameter ``c`` during KDE
            silverman calculations. Default to ``1.06`` if omitted. For more information, please refer to
            references -> Distances with mixed type variables some modified Gower's coefficients (2021) p. 8-9.
        missing_strategy (str): Optional strategy for handling missing values. Can be 'ignore',
            'max_dist' or 'raise_error'. Default is 'ignore' if omitted.
        categorical_ordinal_values_order (dict[int | str, list[str]] | None): Optional dict defining the order of the values contained in
            the columns of type 'categorical_ordinal'. Must contain values for all such columns.
        categorical_ordinal_calculation_type (str): Optional calculation type for categorical
            ordinal features. Can be 'kaufman' or 'podani'. Default is 'kaufman' if omitted.
        k_neighbors (int | None): Optional number of nearest neighbors for 'knn' discretization.
            Default is None if omitted. If k_neighbors is None, it will be set to the square root of the number of points.
        conditional_distances (bool): Default to False. If set to True, two-step approach will be
            triggered to calculate formula. More information in `references year 2021 -> chapter 3 <https://arxiv.org/abs/2101.02481>`_.
        conditional_distances_threshold_coeff (int): Value to be used as the numerator in the fraction (with p_cat as the denominator)
            that defines the threshold above which the distance will be set to 1. More information in reference from year 2021 -> chapter 3.
        handle_unseen_binary_asymmetric (UnseenStrategy): Strategy for handling unseen categories in binary asymmetric features. Can be 'warning', 'error' or 'missing'.
            Default is 'error' if omitted. An unseen value is refused whatever this is set to: an asymmetric feature has only presence and absence (Gower, 1971).
        binary_asymmetric_value_order (dict[int | str, list[object]] | None): Optional explicit ordering of binary values for binary_asymmetric features.
            Similar to categorical_ordinal_values_order, as [absent, present]. If None, only a 0/1 or False/True coding is accepted (1 and True are present).
            If provided, must contain exactly 2 values per binary column. Example: {0: [False, True], 1: ['No', 'Yes']}.
            Recommended for production to ensure reproducibility and handle expected-but-not-yet-seen values.
        handle_unseen_binary_symmetric (UnseenStrategy): Strategy for handling unseen categories in binary symmetric features. Can be 'warning', 'error' or 'missing'.
            Default is 'error' if omitted.
        binary_symmetric_value_order (dict[int | str, list[object]] | None): Optional explicit ordering of binary values for binary_symmetric features.
            Similar to categorical_ordinal_values_order. If None, values are auto-detected from training data.
            If provided, must contain exactly 2 values per binary column. Example: {0: [False, True], 1: ['No', 'Yes']}.
            Recommended for production to ensure reproducibility and handle expected-but-not-yet-seen values.
        handle_unseen_categorical_nominal (UnseenStrategy): Strategy for handling unseen categories in categorical nominal features. Can be 'warning', 'error' or 'missing'.
            Default is 'error' if omitted.
        handle_unseen_categorical_ordinal (UnseenStrategy): Strategy for handling unseen categories in categorical ordinal features. Can be 'warning', 'error' or 'missing'.
            Default is 'error' if omitted.
        out_of_range (OutOfRangeStrategy): Strategy for handling numeric values outside the fitted range. Default is 'error' if omitted.
            Can be: 'clip' silently clips normalized distances to [0, 1], 'warning' emits a UserWarning and clips, 'error' raises ValueError.
        skip_out_of_range_validation (bool): If True, skips validation of out-of-range values.
            Default is ``False`` if omitted.

    Raises:
            ValueError: If custom validation rules fail.

    """

    feature_types: FeatureTypes
    feature_weights: WeightsType | None = {}
    data_type: type[np.floating] | None = np.float32
    scale_method: Literal["range", "iqr"] = "range"
    discretization: Literal["silverman", "knn"] | None = None
    silverman_constant: float = Field(default=1.06, gt=0)
    missing_strategy: Literal["ignore", "max_dist", "raise_error"] = "ignore"
    categorical_ordinal_values_order: OrdinalValueOrders | None = Field(
        default={},
        validate_default=True,
    )
    categorical_ordinal_calculation_type: Literal["kaufman", "podani"] = "kaufman"
    k_neighbors: int | None = Field(default=None, ge=1)
    conditional_distances: bool = False
    conditional_distances_threshold_coeff: int = 1
    handle_unseen_binary_asymmetric: UnseenStrategy = "error"
    binary_asymmetric_value_order: BinaryValueOrders | None = None
    handle_unseen_binary_symmetric: UnseenStrategy = "error"
    binary_symmetric_value_order: BinaryValueOrders | None = None
    handle_unseen_categorical_nominal: UnseenStrategy = "error"
    handle_unseen_categorical_ordinal: UnseenStrategy = "error"
    out_of_range: OutOfRangeStrategy = "error"
    skip_out_of_range_validation: StrictBool = False

    @field_validator("feature_types")
    @classmethod
    def check_feature_types(cls, v: FeatureTypes) -> FeatureTypes:
        """Validate that all provided feature types are supported.

        Args:
            v (FeatureTypes): Mapping of features to their types.

        Returns:
            v (FeatureTypes): The validated mapping, numpy keys made int or str.

        Raises:
            ValueError: If a key is not a str or an integer, or an unsupported
                feature type is encountered.

        """
        valid_types = set(get_args(FeatureType))
        checked: dict[int | str, str] = {}
        for key, val in v.items():
            whole = int(key) if isinstance(key, float) and key.is_integer() else key
            try:
                column = str(whole) if isinstance(whole, str) else operator.index(whole)
            except TypeError:
                msg = f"Invalid key {key!r}: must be a column name (str) or position (int)"
                raise ValueError(msg) from None
            if val not in valid_types:
                msg = (
                    f"Invalid feature type '{val}' for key '{column}'. "
                    f"Must be one of {valid_types}"
                )
                raise ValueError(msg)
            checked[column] = val
        return checked

    @model_validator(mode="after")
    def check_silverman_constant(self) -> "Config":
        """Validate the Silverman constant.

        If the user explicitly sets ``silverman_constant`` verify that ``discretization='silverman'``.
        Also if the user explicitly sets ``k_neighbors`` verify that ``discretization='knn'``.

        Returns:
            Config: The validated config instance.

        Raises:
            UserWarning: If a constant was explicitly set without the required discretization method.

        """
        if (
            "silverman_constant" in self.model_fields_set
            and self.discretization != "silverman"
        ):
            msg = (
                "silverman_constant requires discretization='silverman', "
                f"got discretization={self.discretization!r}"
            )
            raise UserWarning(msg)
        if "k_neighbors" in self.model_fields_set and self.discretization != "knn":
            msg = (
                "k_neighbors require discretization='knn', "
                f"got discretization={self.discretization!r}"
            )
            raise UserWarning(msg)
        return self

    @field_validator("categorical_ordinal_values_order")
    @classmethod
    def check_ordinal_orders(
        cls,
        v: OrdinalValueOrders | None,
        info: ValidationInfo,
    ) -> OrdinalValueOrders | None:
        """Verify that value orders are defined for all categorical ordinal columns.

        Args:
            v (OrdinalValueOrders | None): The order definitions to check.
            info (ValidationInfo): Validation context containing feature types.

        Returns:
            v (OrdinalValueOrders | None): The validated order definitions.

        Raises:
            ValueError: If any categorical ordinal column is missing an order definition.

        """
        if not info.data:  # pragma: no cover
            return v

        ord_cols = {
            k
            for k, t in info.data.get("feature_types", {}).items()
            if t == "categorical_ordinal"
        }
        if ord_cols:
            if not v:
                msg = f"Categorical ordinal columns {ord_cols} must have a values order defined."
                raise ValueError(msg)
            missing = ord_cols - v.keys()
            if missing:
                msg = f"Missing order definitions for columns: {missing}"
                raise ValueError(msg)
            for col_key, order_list in v.items():
                if len(order_list) == 0:
                    msg = f"Empty values order for column '{col_key}'. At least one ordinal level is required."
                    raise ValueError(msg)
        return v

    @field_validator("conditional_distances")
    @classmethod
    def check_conditional_distances(
        cls,
        v: bool,
        info: ValidationInfo,
    ) -> bool:
        """Validate the conditional distances flag and check prerequisites.

        Args:
            v (bool): The flag value to check.
            info (ValidationInfo): Validation context containing feature types.

        Returns:
            v (bool): The validated flag.

        Raises:
            ValueError: If prerequisites (mixed data types) are not met when enabled.

        """
        if v and info.data:  # pragma: no branch
            types = list(info.data.get("feature_types", {}).values())
            first = {"binary_symmetric", "binary_asymmetric", "categorical_nominal"}
            second = {"numeric", "ratio_scale_interval", "categorical_ordinal"}
            if not any(t in first for t in types) or not any(
                t in second for t in types
            ):
                msg = (
                    "For conditional distances at least one binary or "
                    "categorical_nominal feature (first step) and at least one "
                    "numeric, ratio_scale_interval or categorical_ordinal "
                    "feature (second step) must be present"
                )
                raise ValueError(msg)
        return v

    @field_validator("feature_weights")
    @classmethod
    def check_feature_weights(
        cls,
        v: "WeightsType | None",
    ) -> "WeightsType | None":
        """Reject negative feature weights.

        Args:
            v (WeightsType | None): The weights to check.

        Returns:
            v (WeightsType | None): The validated weights.

        Raises:
            ValueError: If any weight is negative or not finite.

        """
        if not isinstance(v, Mapping):
            return v

        bad = {k: w for k, w in v.items() if not math.isfinite(w) or w < 0}
        if bad:
            msg = (
                "feature_weights must be finite and non-negative; "
                f"got {bad}. Gower's coefficient is only well behaved for "
                "w_k >= 0, and negative weights can produce negative distances."
            )
            raise ValueError(msg)
        return v

    @field_validator("conditional_distances_threshold_coeff")
    @classmethod
    def check_threshold_coeff(cls, v: int) -> int:
        """Validate the threshold coefficient for conditional distances.

        Args:
            v (int): The coefficient to check.

        Returns:
            v (int): The validated coefficient.

        Raises:
            ValueError: If the coefficient is less than 1.

        """
        if v < 1:
            msg = f"conditional_distances_threshold_coeff must be at least 1, got {v}"
            raise ValueError(msg)
        return v

    @staticmethod
    def _check_binary_value_order(
        v: BinaryValueOrders | None,
        info: ValidationInfo,
        binary_type: str,
    ) -> BinaryValueOrders | None:
        """Verify that binary value orders contain exactly 2 unique values per column.

        Args:
            v (BinaryValueOrders | None): The binary value order definitions.
            info (ValidationInfo): Validation context containing feature types.
            binary_type (str): The binary feature type name (e.g. 'binary_asymmetric').

        Returns:
            BinaryValueOrders | None: The validated order definitions.

        Raises:
            ValueError: If any binary column order doesn't have exactly 2 values.

        """
        if v is None:
            return v

        if not info.data:  # pragma: no cover
            return v

        expected_binary_values = 2

        binary_cols = {
            k for k, t in info.data.get("feature_types", {}).items() if t == binary_type
        }

        for col_idx, values in v.items():
            if len(values) != expected_binary_values:
                msg = (
                    f"Binary {binary_type} column {col_idx} must have exactly {expected_binary_values} values in order, "
                    f"got {len(values)}: {values}"
                )
                raise ValueError(msg)

            if len(set(values)) != expected_binary_values:
                msg = f"Binary {binary_type} values for column {col_idx} must be unique, got {values}"
                raise ValueError(msg)

        extra_cols = set(v.keys()) - binary_cols if binary_cols else set(v.keys())
        if extra_cols:
            msg = (
                f"{binary_type}_value_order contains non-{binary_type} columns: {extra_cols}. "
                f"Binary {binary_type} columns are: {binary_cols or 'none'}"
            )
            raise ValueError(msg)

        return v

    @field_validator("binary_asymmetric_value_order")
    @classmethod
    def check_binary_asymmetric_value_order(
        cls,
        v: BinaryValueOrders | None,
        info: ValidationInfo,
    ) -> BinaryValueOrders | None:
        """Validate binary asymmetric value order definitions."""
        return cls._check_binary_value_order(v, info, "binary_asymmetric")

    @field_validator("binary_symmetric_value_order")
    @classmethod
    def check_binary_symmetric_value_order(
        cls,
        v: BinaryValueOrders | None,
        info: ValidationInfo,
    ) -> BinaryValueOrders | None:
        """Validate binary symmetric value order definitions."""
        return cls._check_binary_value_order(v, info, "binary_symmetric")
