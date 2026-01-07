from typing import Literal, get_args

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
ScaleMethod = Literal["range", "iqr"]
ScaleWindow = Literal["kde", "kNN"]
ScaleWindowType = Literal["silverman"]
MissingStrategy = Literal["ignore", "max_dist", "raise_error"]
CategoricalOrdinalCalcType = Literal["kaufman", "podani"]
K_NeighboursType = int | None
ConditionalDistancesFlag = Literal[True, False]
ConditionalDistancesThresholdCoeffType = int


class Config(BaseModel):
    """Configuration object to initialize Gower.

    Args:
        feature_types (dict[int | str, str]): Mapping of column indices (or DataFrame column names) to
            specific type.
        feature_weights (dict[int, float] | str | None): Optional mapping of column indices (or names) to a float weight.
            If None or "uniform", all features will have equal weight of 1. Otherwise,
            the weights must be a dictionary mapping feature indices to weights, i.e.
            {0: 1.0, 1: 2.0}.
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
        k_neighbours (int | None): Optional number of nearest neighbors for 'kNN' scaling window.
            Default is None if omitted. If k_neighbours is None, it will be set to the square root of the number of points.
        conditional_distances (bool): Default to False. If set to True, two-step approach will be
            triggered to calculate formula. More information in `references year 2021 -> chapter 3 <https://arxiv.org/abs/2101.02481>`_.
        conditional_distances_threshold_coeff (int): Value to be used as the numerator in the fraction (with p_cat as the denominator)
            that defines the threshold above which the distance will be set to 1. More information in reference from year 2021 -> chapter 3.


    Raises:
            ValueError: If custom validation rule fail.

    """

    feature_types: dict[int | str, str]
    feature_weights: WeightsType | None = {}
    scale_method: ScaleMethod = "range"
    scale_window: ScaleWindow | None = None
    scale_window_type: ScaleWindowType | None = None
    missing_strategy: MissingStrategy = "ignore"
    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {}
    categorical_ordinal_calculation_type: CategoricalOrdinalCalcType = "kaufman"
    k_neighbours: int | None = None
    conditional_distances: ConditionalDistancesFlag = False
    conditional_distances_threshold_coeff: int = 1

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

    @field_validator("feature_weights")
    @classmethod
    def check_weights(
        cls,
        v: WeightsType | None,
    ) -> WeightsType | None:
        """Validate the feature weights configuration.

        Args:
            v (WeightsType | None): The weights configuration to check.

        Returns:
            v (WeightsType | None): The validated weights.

        Raises:
            ValueError: If weights are not 'uniform', a dictionary, or None.

        """
        if v is not None:
            if isinstance(v, str) and v != "uniform":
                msg = f"weights must be 'uniform' or a dict, got {v!r}"
                raise ValueError(msg)
            if not isinstance(v, (str, dict)):
                msg = f"weights must be str, dict, or None, got {type(v).__name__}"
                raise ValueError(msg)
        return v

    @field_validator("scale_method")
    @classmethod
    def check_scale_method(cls, v: ScaleMethod) -> ScaleMethod:
        """Validate the scaling method for numeric features.

        Args:
            v (ScaleMethod): The scaling method to check.

        Returns:
            v (ScaleMethod): The validated scaling method.

        Raises:
            ValueError: If the scaling method is not supported.

        """
        valid_methods = set(get_args(ScaleMethod))
        if v not in valid_methods:
            msg = f"scale_method must be one of {valid_methods}, got {v!r}"
            raise ValueError(msg)
        return v

    @field_validator("scale_window")
    @classmethod
    def check_scale_window(cls, v: ScaleWindow | None) -> ScaleWindow | None:
        """Validate the scaling window method.

        Args:
            v (ScaleWindow | None): The scaling window to check.

        Returns:
            v (ScaleWindow | None): The validated scaling window.

        Raises:
            ValueError: If the scaling window is not supported.

        """
        if v is None:
            return v
        valid_methods = set(get_args(ScaleWindow))
        if v not in valid_methods:
            msg = f"scale_window must be one of {valid_methods}, got {v!r}"
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

    @field_validator("missing_strategy")
    @classmethod
    def check_missing_strategy(
        cls,
        v: MissingStrategy,
    ) -> MissingStrategy:
        """Validate the strategy for handling missing values.

        Args:
            v (MissingStrategy): The strategy to check.

        Returns:
            v (MissingStrategy): The validated strategy.

        Raises:
            ValueError: If the strategy is not supported.

        """
        valid_strategies = set(get_args(MissingStrategy))

        if v not in valid_strategies:
            msg = f"missing_strategy must be one of {valid_strategies}, got {v!r}"
            raise ValueError(msg)
        return v

    @field_validator("k_neighbours")
    @classmethod
    def check_k_neighbours(
        cls,
        v: K_NeighboursType,
    ) -> K_NeighboursType:
        """Validate the number of nearest neighbors (k).

        Args:
            v (K_NeighboursType): The value of k to check.

        Returns:
            v (K_NeighboursType): The validated value.

        Raises:
            ValueError: If k is not a positive integer or None.

        """
        if v is not None and (not isinstance(v, int) or v < 1):
            msg = f"k_neighbours must be None or a positive integer, got {v!r}"
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

    @field_validator("categorical_ordinal_calculation_type")
    @classmethod
    def check_ordinal_calc_type(
        cls,
        v: CategoricalOrdinalCalcType,
    ) -> CategoricalOrdinalCalcType:
        """Validate the calculation method for categorical ordinal features.

        Args:
            v (CategoricalOrdinalCalcType): The calculation method to check.

        Returns:
            v (CategoricalOrdinalCalcType): The validated method.

        Raises:
            ValueError: If the calculation method is not supported.

        """
        valid_types = set(get_args(CategoricalOrdinalCalcType))
        if v not in valid_types:
            msg = f"categorical_ordinal_calculation_type must be one of {valid_types}, got {v!r}"
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
