from typing import Any, Literal, get_args

import numpy as np
from pydantic import BaseModel, ValidationInfo, field_validator

FeatureType = Literal[
    "numeric",
    "categorical_nominal",
    "categorical_ordinal",
    "binary_asymmetric",
    "binary_symmetric",
    "ratio_scale_interval",
]
WeightsType = Literal["uniform"] | dict[int, float]
DataType = type[np.integer] | type[np.floating]
ScaleMethod = Literal["range", "iqr"]
ScaleWindow = Literal["kde", "kNN"]
ScaleWindowType = Literal["silverman"]
MissingStrategy = Literal["ignore", "max_dist", "raise_error"]
CategoricalOrdinalCalcType = Literal["kaufman", "podani"]
K_NeighborsType = int | None
ConditionalDistancesFlag = Literal[True, False]
ConditionalDistancesThresholdCoeffType = int
HandleUnseenBinaryAsymmetric = Literal["warning", "error", "missing"]
BinaryAsymmetricValueOrderType = dict[int | str, list[Any]] | None
HandleUnseenBinarySymmetric = Literal["warning", "error", "missing"]
BinarySymmetricValueOrderType = dict[int | str, list[Any]] | None
HandleUnseenCategoricalNominal = Literal["warning", "error", "missing"]
HandleUnseenCategoricalOrdinal = Literal["warning", "error", "missing"]


class Config(BaseModel):
    """Configuration object to initialize Gower.

    Args:
        feature_types (dict[int | str, str]): Mapping of column indices (or DataFrame column names) to
            specific type.
        feature_weights (dict[int, float] | str | None): Optional mapping of column indices (or names) to a float weight.
            If None or "uniform", all features will have equal weight of 1. Otherwise,
            the weights must be a dictionary mapping feature indices to weights, i.e.
            {0: 1.0, 1: 2.0}.
        data_type (DataType): Optional data type required for __call__, transformation and matrix method.
            If omitted, np.float32 will be used.
        scale_method (str): Optional scaling method for numeric features. Can be 'range' or 'iqr'.
            Default is 'range' if omitted.
        scale_window (str | None): Optional scaling window for numeric or ratio features. Can be None, 'kde'
            or 'kNN'. Default is None if omitted.
        scale_window_type (str | None): Optional type of scaling window. Can be None or 'silverman'.
            Default is None if omitted, not recommended to use without scale_window.
        missing_strategy (str): Optional strategy for handling missing values. Can be 'ignore',
            'max_dist' or 'raise_error'. Default is 'ignore' if omitted.
        categorical_ordinal_values_order (dict[int | str, list[str]] | None): Optional dict defining the order of the values contained in
            the columns of type 'categorical_ordinal'. Must contain values for all such columns.
        categorical_ordinal_calculation_type (str): Optional calculation type for categorical
            ordinal features. Can be 'kaufman' or 'podani'. Default is 'kaufman' if omitted.
        k_neighbors (int | None): Optional number of nearest neighbors for 'kNN' scaling window.
            Default is None if omitted. If k_neighbors is None, it will be set to the square root of the number of points.
        conditional_distances (bool): Default to False. If set to True, two-step approach will be
            triggered to calculate formula. More information in `references year 2021 -> chapter 3 <https://arxiv.org/abs/2101.02481>`_.
        conditional_distances_threshold_coeff (int): Value to be used as the numerator in the fraction (with p_cat as the denominator)
            that defines the threshold above which the distance will be set to 1. More information in reference from year 2021 -> chapter 3.
        handle_unseen_binary_asymmetric (str): Strategy for handling unseen categories in binary asymmetric features. Can be 'warning', 'error' or 'missing'.
            Default is 'error' if omitted.
        binary_asymmetric_value_order (dict[int | str, list[Any]] | None): Optional explicit ordering of binary values for binary_asymmetric features.
            Similar to categorical_ordinal_values_order. If None, values are auto-detected from training data.
            If provided, must contain exactly 2 values per binary column. Example: {0: [False, True], 1: ['No', 'Yes']}.
            Recommended for production to ensure reproducibility and handle expected-but-not-yet-seen values.
        handle_unseen_binary_symmetric (str): Strategy for handling unseen categories in binary symmetric features. Can be 'warning', 'error' or 'missing'.
            Default is 'error' if omitted.
        binary_symmetric_value_order (dict[int | str, list[Any]] | None): Optional explicit ordering of binary values for binary_symmetric features.
            Similar to categorical_ordinal_values_order. If None, values are auto-detected from training data.
            If provided, must contain exactly 2 values per binary column. Example: {0: [False, True], 1: ['No', 'Yes']}.
            Recommended for production to ensure reproducibility and handle expected-but-not-yet-seen values.
        handle_unseen_categorical_nominal (str): Strategy for handling unseen categories in categorical nominal features. Can be 'warning', 'error' or 'missing'.
            Default is 'error' if omitted.
        handle_unseen_categorical_ordinal (str): Strategy for handling unseen categories in categorical ordinal features. Can be 'warning', 'error' or 'missing'.
            Default is 'error' if omitted.

    Raises:
            ValueError: If custom validation rule fail.

    """

    feature_types: dict[int | str, str]
    feature_weights: WeightsType | None = {}
    data_type: DataType | None = np.float32
    scale_method: ScaleMethod = "range"
    scale_window: ScaleWindow | None = None
    scale_window_type: ScaleWindowType | None = None
    missing_strategy: MissingStrategy = "ignore"
    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {}
    categorical_ordinal_calculation_type: CategoricalOrdinalCalcType = "kaufman"
    k_neighbors: int | None = None
    conditional_distances: ConditionalDistancesFlag = False
    conditional_distances_threshold_coeff: int = 1
    handle_unseen_binary_asymmetric: HandleUnseenBinaryAsymmetric = "error"
    binary_asymmetric_value_order: BinaryAsymmetricValueOrderType = None
    handle_unseen_binary_symmetric: HandleUnseenBinarySymmetric = "error"
    binary_symmetric_value_order: BinarySymmetricValueOrderType = None
    handle_unseen_categorical_nominal: HandleUnseenCategoricalNominal = "error"
    handle_unseen_categorical_ordinal: HandleUnseenCategoricalOrdinal = "error"

    @field_validator("feature_types")
    @classmethod
    def check_feature_types(cls, v: dict[int | str, str]) -> dict[int | str, str]:
        """Validate that all provided feature types are supported.

        Args:
            v (dict[int | str, str]): Dictionary mapping features to their types.

        Returns:
            v (dict[int | str, str]): The validated dictionary.

        Raises:
            ValueError: If an unsupported feature type is encountered.

        """
        valid_types = set(get_args(FeatureType))
        for key, val in v.items():
            if val not in valid_types:
                msg = (
                    f"Invalid feature type '{val}' for key '{key}'. "
                    f"Must be one of {valid_types}"
                )
                raise ValueError(msg)
        return v

    @field_validator("scale_window_type")
    @classmethod
    def check_scale_window_type(
        cls,
        v: ScaleWindowType | None,
        info: ValidationInfo,
    ) -> ScaleWindowType | None:
        """Validate compatibility between scale_window_type and scale_window.

        Args:
            v (ScaleWindowType | None): The window type to check.
            info (ValidationInfo): Validation context containing other fields.

        Returns:
            v (ScaleWindowType | None): The validated window type.

        Raises:
            ValueError: If the window type is incompatible with the selected scale_window.

        """
        if not info.data:
            return v

        scale_window = info.data.get("scale_window")
        if scale_window is None:
            if v is not None:
                msg = "scale_window_type must be None when scale_window is None"
                raise ValueError(msg)
        elif scale_window == "kde" and v not in (None, "silverman"):
            msg = "scale_window_type must be one of [None, 'silverman'] when scale_window='kde'"
            raise ValueError(msg)
        return v

    @field_validator("k_neighbors")
    @classmethod
    def check_k_neighbors(
        cls,
        v: K_NeighborsType,
    ) -> K_NeighborsType:
        """Validate the number of nearest neighbors (k).

        Args:
            v (K_NeighborsType): The value of k to check.

        Returns:
            v (K_NeighborsType): The validated value.

        Raises:
            ValueError: If k is not a positive integer or None.

        """
        if v is not None and (not isinstance(v, int) or v < 1):
            msg = f"k_neighbors must be None or a positive integer, got {v!r}"
            raise ValueError(msg)
        return v

    @field_validator("categorical_ordinal_values_order")
    @classmethod
    def check_ordinal_orders(
        cls,
        v: dict[int | str, list[str]] | None,
        info: ValidationInfo,
    ) -> dict[int | str, list[str]] | None:
        """Verify that value orders are defined for all categorical ordinal columns.

        Args:
            v (dict[int | str, list[str]] | None): The order definitions to check.
            info (ValidationInfo): Validation context containing feature types.

        Returns:
            v (dict[int | str, list[str]] | None): The validated order definitions.

        Raises:
            ValueError: If any categorical ordinal column is missing an order definition.

        """
        if not info.data:
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
        return v

    @field_validator("conditional_distances")
    @classmethod
    def check_conditional_distances(
        cls,
        v: ConditionalDistancesFlag,
        info: ValidationInfo,
    ) -> ConditionalDistancesFlag:
        """Validate the conditional distances flag and check prerequisites.

        Args:
            v (ConditionalDistancesFlag): The flag value to check.
            info (ValidationInfo): Validation context containing feature types.

        Returns:
            v (ConditionalDistancesFlag): The validated flag.

        Raises:
            ValueError: If prerequisites (mixed data types) are not met when enabled.

        """
        if v not in (True, False):
            msg = f"conditional_distances must be True or False, got {v}"
            raise ValueError(msg)

        if v and info.data:
            feature_types = info.data.get("feature_types", {})
            n_feats = len(feature_types)
            n_num_feats = sum(
                1
                for t in feature_types.values()
                if t in ("numeric", "ratio_scale_interval")
            )
            if n_num_feats in (0, n_feats):
                msg = "For conditional distances both categorical and numerical features must be present"
                raise ValueError(msg)
        return v

    @field_validator("conditional_distances_threshold_coeff")
    @classmethod
    def check_threshold_coeff(
        cls,
        v: ConditionalDistancesThresholdCoeffType,
    ) -> ConditionalDistancesThresholdCoeffType:
        """Validate the threshold coefficient for conditional distances.

        Args:
            v (ConditionalDistancesThresholdCoeffType): The coefficient to check.

        Returns:
            v (ConditionalDistancesThresholdCoeffType): The validated coefficient.

        Raises:
            ValueError: If the coefficient is less than 1.

        """
        if v < 1:
            msg = f"conditional_distances_threshold_coeff must be at least 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("handle_unseen_binary_asymmetric")
    @classmethod
    def check_handle_unseen_binary_asymmetric(
        cls,
        v: HandleUnseenBinaryAsymmetric,
    ) -> HandleUnseenBinaryAsymmetric:
        """Validate the strategy for handling unseen categories in binary asymmetric features.

        Args:
            v (HandleUnseenBinaryAsymmetric): The strategy to check.

        Returns:
            v (HandleUnseenBinaryAsymmetric): The validated strategy.

        Raises:
            ValueError: If the strategy is not one of 'warning', 'error', or 'missing'.

        """
        valid_strategies = get_args(HandleUnseenBinaryAsymmetric)
        if v not in valid_strategies:
            msg = f"handle_unseen_binary_asymmetric must be one of {valid_strategies}, got {v}"
            raise ValueError(msg)

        return v

    @field_validator("binary_asymmetric_value_order")
    @classmethod
    def check_binary_asymmetric_value_order(
        cls,
        v: BinaryAsymmetricValueOrderType,
        info: ValidationInfo,
    ) -> BinaryAsymmetricValueOrderType:
        """Verify that binary value orders contain exactly 2 values per column.

        Args:
            v (BinaryAsymmetricValueOrderType): The binary value order definitions.
            info (ValidationInfo): Validation context containing feature types.

        Returns:
            BinaryAsymmetricValueOrderType: The validated order definitions.

        Raises:
            ValueError: If any binary column order doesn't have exactly 2 values.
            TypeError: If any binary column order is not a list.

        """
        if v is None:
            return v

        if not info.data:
            return v

        expected_binary_values = 2

        binary_asymmetric_cols = {
            k
            for k, t in info.data.get("feature_types", {}).items()
            if t == "binary_asymmetric"
        }

        for col_idx, values in v.items():
            if not isinstance(values, list):
                msg = f"Binary values for column {col_idx} must be a list, got {type(values)}"
                raise TypeError(msg)

            if len(values) != expected_binary_values:
                msg = (
                    f"Binary asymmetric column {col_idx} must have exactly {expected_binary_values} values in order, "
                    f"got {len(values)}: {values}"
                )
                raise ValueError(msg)

            if len(set(values)) != expected_binary_values:
                msg = f"Binary asymmetric values for column {col_idx} must be unique, got {values}"
                raise ValueError(msg)

        extra_cols = (
            set(v.keys()) - binary_asymmetric_cols
            if binary_asymmetric_cols
            else set(v.keys())
        )
        if extra_cols:
            msg = (
                f"binary_asymmetric_value_order contains non-binary_asymmetric columns: {extra_cols}. "
                f"Binary asymmetric columns are: {binary_asymmetric_cols or 'none'}"
            )
            raise ValueError(msg)

        return v

    @field_validator("handle_unseen_binary_symmetric")
    @classmethod
    def check_handle_unseen_binary_symmetric(
        cls,
        v: HandleUnseenBinarySymmetric,
    ) -> HandleUnseenBinarySymmetric:
        """Validate the strategy for handling unseen categories in binary symmetric features.

        Args:
            v (HandleUnseenBinarySymmetric): The strategy to check.

        Returns:
            v (HandleUnseenBinarySymmetric): The validated strategy.

        Raises:
            ValueError: If the strategy is not one of 'warning', 'error', or 'missing'.

        """
        valid_strategies = get_args(HandleUnseenBinarySymmetric)
        if v not in valid_strategies:
            msg = f"handle_unseen_binary_symmetric must be one of {valid_strategies}, got {v}"
            raise ValueError(msg)

        return v

    @field_validator("binary_symmetric_value_order")
    @classmethod
    def check_binary_symmetric_value_order(
        cls,
        v: BinarySymmetricValueOrderType,
        info: ValidationInfo,
    ) -> BinarySymmetricValueOrderType:
        """Verify that binary symmetric value orders contain exactly 2 values per column.

        Args:
            v (BinarySymmetricValueOrderType): The binary value order definitions.
            info (ValidationInfo): Validation context containing feature types.

        Returns:
            BinarySymmetricValueOrderType: The validated order definitions.

        Raises:
            ValueError: If any binary column order doesn't have exactly 2 values.
            TypeError: If any binary column order is not a list.

        """
        if v is None:
            return v

        if not info.data:
            return v

        expected_binary_values = 2

        binary_symmetric_cols = {
            k
            for k, t in info.data.get("feature_types", {}).items()
            if t == "binary_symmetric"
        }

        for col_idx, values in v.items():
            if not isinstance(values, list):
                msg = f"Binary values for column {col_idx} must be a list, got {type(values)}"
                raise TypeError(msg)

            if len(values) != expected_binary_values:
                msg = (
                    f"Binary symmetric column {col_idx} must have exactly {expected_binary_values} values in order, "
                    f"got {len(values)}: {values}"
                )
                raise ValueError(msg)

            if len(set(values)) != expected_binary_values:
                msg = f"Binary symmetric values for column {col_idx} must be unique, got {values}"
                raise ValueError(msg)

        extra_cols = (
            set(v.keys()) - binary_symmetric_cols
            if binary_symmetric_cols
            else set(v.keys())
        )
        if extra_cols:
            msg = (
                f"binary_symmetric_value_order contains non-binary_symmetric columns: {extra_cols}. "
                f"Binary symmetric columns are: {binary_symmetric_cols or 'none'}"
            )
            raise ValueError(msg)

        return v

    @field_validator("handle_unseen_categorical_nominal")
    @classmethod
    def check_handle_unseen_categorical_nominal(
        cls,
        v: HandleUnseenCategoricalNominal,
    ) -> HandleUnseenCategoricalNominal:
        """Validate the strategy for handling unseen categories in categorical nominal features.

        Args:
            v (HandleUnseenCategoricalNominal): The strategy to check.

        Returns:
            v (HandleUnseenCategoricalNominal): The validated strategy.

        Raises:
            ValueError: If the strategy is not one of 'warning', 'error', or 'missing'.

        """
        valid_strategies = get_args(HandleUnseenCategoricalNominal)
        if v not in valid_strategies:
            msg = f"handle_unseen_categorical_nominal must be one of {valid_strategies}, got {v}"
            raise ValueError(msg)

        return v

    @field_validator("handle_unseen_categorical_ordinal")
    @classmethod
    def check_handle_unseen_categorical_ordinal(
        cls,
        v: HandleUnseenCategoricalOrdinal,
    ) -> HandleUnseenCategoricalOrdinal:
        """Validate the strategy for handling unseen categories in categorical ordinal features.

        Args:
            v (HandleUnseenCategoricalOrdinal): The strategy to check.

        Returns:
            v (HandleUnseenCategoricalOrdinal): The validated strategy.

        Raises:
            ValueError: If the strategy is not one of 'warning', 'error', or 'missing'.

        """
        valid_strategies = get_args(HandleUnseenCategoricalOrdinal)
        if v not in valid_strategies:
            msg = f"handle_unseen_categorical_ordinal must be one of {valid_strategies}, got {v}"
            raise ValueError(msg)

        return v
