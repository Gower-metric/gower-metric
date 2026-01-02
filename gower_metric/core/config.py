from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, field_validator

# Allowed values as Literals
FeatureType = Literal[
    "numeric",
    "categorical_nominal",
    "categorical_ordinal",
    "binary_asymmetric",
    "binary_symmetric",
    "ratio_scale_interval",
]

ScaleMethod = Literal["range", "iqr"]
ScaleWindow = Literal["kde", "kNN"]
ScaleWindowType = Literal["silverman"]
MissingStrategy = Literal["ignore", "max_dist", "raise_error"]
CategoricalOrdinalCalcType = Literal["kaufman", "podani"]
WeightsType = Literal["uniform", None]
ConditionalDistancesFlag = Literal[True, False]


class Config(BaseModel):
    """
    Configuration object to initialize Gower.

    Args:
        feature_types (dict[int | str, str]): Mapping of column indices (or DataFrame column names) to
            specific type.
        feature_weights (dict[int, float] | str | None): Optional mapping of column indices (or names) to a float weight.
            If None or "uniform", all features will have equal weight of 1. Otherwise,
            the weights must be a dictionary mapping feature indices to weights, i.e.
            {0: 1.0, 1: 2.0}.
        scale_method (str): Optional scaling method for numeric features. Can be 'range' or 'iqr'.
            Default is 'range' if omitted.
        scale_window (Optional[str]): Optional scaling window for numeric or ratio features. Can be None, 'kde'
            or 'kNN'. Default is None if omitted.
        scale_window_type (Optional[str]): Optional type of scaling window. Can be None or 'silverman'.
            Default is None if omitted, not recommended to use without scale_window.
        missing_strategy (str): Optional strategy for handling missing values. Can be 'ignore',
            'max_dist' or 'raise_error'. Default is 'ignore' if omitted.
        categorical_ordinal_values_order (dict[int | str, list[str]] | None): Optional dict defining the order of the values contained in
            the columns of type 'categorical_ordinal'. Must contain values for all such columns.
        categorical_ordinal_calculation_type (str): Optional calculation type for categorical
            ordinal features. Can be 'kaufman' or 'podani'. Default is 'kaufman' if omitted.
        k_neighbours (Optional[int]): Optional number of nearest neighbors for 'kNN' scaling window.
            Default is None if omitted. If k_neighbours is None or less than 1, it will be
            set to the square root of the number of points.
        conditional_distances (bool): Default to False. If set to True, two-step approach will be
            triggered to calculate formula. More information in references -> chapter 3.
        conditional_distances_threshold_coeff (int): Value to be used as the numerator in the fraction (with p_cat as the denominator)
            that defines the threshold above which the distance will be set to 1. More information in references -> chapter 3.


        Raises:
            ValueError: If custom validation rule fail.
    """

    feature_types: Dict[int | str, FeatureType]
    feature_weights: Optional[Union[str, Dict[int | str, float]]] = {}
    scale_method: ScaleMethod = "range"
    scale_window: Optional[ScaleWindow] = None
    scale_window_type: Optional[ScaleWindowType] = None
    missing_strategy: MissingStrategy = "ignore"
    categorical_ordinal_values_order: Optional[Dict[int | str, List[str]]] = {}
    categorical_ordinal_calculation_type: Optional[CategoricalOrdinalCalcType] = "kaufman"
    k_neighbours: Optional[int] = None
    conditional_distances: ConditionalDistancesFlag = False
    conditional_distances_threshold_coeff: int = 1

    @field_validator("categorical_ordinal_values_order")
    def check_ordinal_orders(cls, v, values):
        ord_cols = {k for k, t in values.data.get("feature_types", {}).items() if t == "categorical_ordinal"}
        if ord_cols:
            if not v:
                raise ValueError(f"Categorical ordinal columns {ord_cols} must have a values order defined.")
            missing = ord_cols - v.keys()
            if missing:
                raise ValueError(f"Missing order definitions for columns: {missing}")
        return v

    @field_validator("scale_window_type")
    def check_scale_window_type(cls, v, values):
        scale_window = values.data.get("scale_window")
        if scale_window is None:
            if v is not None:
                raise ValueError("scale_window_type must be None when scale_window is None")
        elif scale_window == "kde" and v not in (None, "silverman"):
            raise ValueError(f"scale_window_type must be one of [None, 'silverman'] when scale_window='kde'")
        return v

    @field_validator("feature_weights")
    def check_weights(cls, v):
        if v is not None:
            if isinstance(v, str) and v != "uniform":
                raise ValueError(f"weights must be 'uniform' or a dict, got {v!r}")
            elif not isinstance(v, (str, dict)):
                raise ValueError(f"weights must be str, dict, or None, got {type(v).__name__}")
        return v

    @field_validator("conditional_distances")
    def check_conditional_distances(cls, v, values):
        print(v)
        if v not in (True, False):
            raise ValueError(f"conditional_distances must be True or False, got {v}")

        if v:
            feature_types = values.data.get("feature_types", {})
            n_feats = len(feature_types)
            n_num_feats = sum(1 for t in feature_types.values() if t in ("numeric", "ratio_scale_interval"))
            if n_num_feats == 0 or n_feats == n_num_feats:
                raise ValueError("For conditional distances both categorical and numerical features must be present")
        return v

    @field_validator("conditional_distances_threshold_coeff")
    def check_threshold_coeff(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"conditional_distances_threshold_coeff must be at least 1, got {v}"
            )
        return v
