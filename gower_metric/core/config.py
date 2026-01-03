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

ScaleMethod = Literal["range", "iqr"]
ScaleWindow = Literal["kde", "kNN"] | None
ScaleWindowType = Literal["silverman"] | None
MissingStrategy = Literal["ignore", "max_dist", "raise_error"]
CategoricalOrdinalCalcType = Literal["kaufman", "podani"]
WeightsType = Literal["uniform"] | None
ConditionalDistancesFlag = Literal[True, False]


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
            Default is None if omitted. If k_neighbours is None or less than 1, it will be
            set to the square root of the number of points.
        conditional_distances (bool): Default to False. If set to True, two-step approach will be
            triggered to calculate formula. More information in references -> chapter 3.
        conditional_distances_threshold_coeff (int): Value to be used as the numerator in the fraction (with p_cat as the denominator)
            that defines the threshold above which the distance will be set to 1. More information in reference from year 2021 -> chapter 3.


    Raises:
            ValueError: If custom validation rule fail.

    """

    feature_types: dict[int | str, str]
    feature_weights: WeightsType | dict[int, float] = {}
    scale_method: ScaleMethod = "range"
    scale_window: ScaleWindow = None
    scale_window_type: ScaleWindowType = None
    missing_strategy: MissingStrategy = "ignore"
    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {}
    categorical_ordinal_calculation_type: CategoricalOrdinalCalcType = "kaufman"
    k_neighbours: int | None = None
    conditional_distances: ConditionalDistancesFlag = False
    conditional_distances_threshold_coeff: int = 1

    @field_validator("feature_types")
    def check_feature_types(self, v: dict[int | str, str]) -> dict[int | str, str]:
        """Ensure that all feature types are valid.

        Args:
            v (dict[int | str, str]): The feature_types value to validate.

        Returns:
            v (dict[int | str, str]): The validated feature_types.

        Raises:
            ValueError: If any feature type is invalid.

        """
        valid_types = set(get_args(FeatureType))
        for key, val in v.items():
            if val not in valid_types:
                msg = (
                    f"Invalid feature type '{val}' for key '{key}'. "
                    f"Must be one of {valid_types}"
                )
                raise ValueError(
                    msg,
                )
        return v

    @field_validator("categorical_ordinal_values_order")
    def check_ordinal_orders(
        self,
        v: dict[int | str, list[str]] | None,
        info: ValidationInfo,
    ) -> dict[int | str, list[str]] | None:
        """Ensure that all categorical ordinal columns have an order defined.

        Args:
            v (dict[int | str, list[str]] | None): The categorical_ordinal_values_order value to validate.
            info (ValidationInfo): The validation info containing other fields.

        Returns:
            v (dict[int | str, list[str]] | None): The validated categorical_ordinal_values_order.

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
                raise ValueError(
                    msg,
                )
            missing = ord_cols - v.keys()
            if missing:
                msg = f"Missing order definitions for columns: {missing}"
                raise ValueError(msg)
        return v

    @field_validator("scale_window_type")
    def check_scale_window_type(
        self,
        v: ScaleWindowType | None,
        info: ValidationInfo,
    ) -> ScaleWindowType | None:
        """Ensure that scale_window_type is compatible with scale_window.

        Args:
            v (ScaleWindowType | None): The scale_window_type value to validate.
            info (ValidationInfo): The validation info containing other fields.

        Returns:
            v (ScaleWindowType | None): The validated scale_window_type.

        Raises:
            ValueError: If scale_window_type is incompatible with scale_window.

        """
        if not info.data:
            return v

        scale_window = info.data.get("scale_window")
        if scale_window is None:
            if v is not None:
                msg = "scale_window_type must be None when scale_window is None"
                raise ValueError(
                    msg,
                )
        elif scale_window == "kde" and v not in (None, "silverman"):
            msg = "scale_window_type must be one of [None, 'silverman'] when scale_window='kde'"
            raise ValueError(
                msg,
            )
        return v

    @field_validator("feature_weights")
    def check_weights(
        self,
        v: WeightsType | dict[int, float],
    ) -> WeightsType | dict[int, float]:
        """Ensure that feature_weights is either 'uniform' or a dict.

        Args:
            v (WeightsType | dict[int | str, float]): The feature_weights value to validate.

        Returns:
            v (WeightsType | dict[int | str, float]): The validated feature_weights.

        Raises:
            ValueError: If feature_weights is not 'uniform' or a dict.

        """
        if v is not None:
            if isinstance(v, str) and v != "uniform":
                msg = f"weights must be 'uniform' or a dict, got {v!r}"
                raise ValueError(msg)
            if not isinstance(v, (str, dict)):
                msg = f"weights must be str, dict, or None, got {type(v).__name__}"
                raise ValueError(
                    msg,
                )
        return v

    @field_validator("conditional_distances")
    def check_conditional_distances(
        self,
        v: ConditionalDistancesFlag,
        info: ValidationInfo,
    ) -> ConditionalDistancesFlag:
        """Ensure that conditional_distances is a boolean and that both categorical and numerical features are present if True.

        Args:
            v (ConditionalDistancesFlag): The conditional_distances value to validate.
            info (ValidationInfo): The validation info containing other fields.

        Returns:
            v (ConditionalDistancesFlag): The validated conditional_distances.

        Raises:
            ValueError: If conditional_distances is not a boolean or if both categorical and numerical features are not present when True.

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
                raise ValueError(
                    msg,
                )
        return v

    @field_validator("conditional_distances_threshold_coeff")
    def check_threshold_coeff(self, v: int) -> int:
        """Ensure that conditional_distances_threshold_coeff is at least 1.

        Args:
            v (int): The conditional_distances_threshold_coeff value to validate.

        Returns:
            v (int): The validated conditional_distances_threshold_coeff.

        Raises:
            ValueError: If conditional_distances_threshold_coeff is less than 1.

        """
        if v < 1:
            msg = f"conditional_distances_threshold_coeff must be at least 1, got {v}"
            raise ValueError(
                msg,
            )
        return v
