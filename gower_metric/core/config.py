from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator

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
    feature_types: Dict[int | str, FeatureType]
    scale_method: ScaleMethod = "range"
    scale_window: Optional[ScaleWindow] = None
    scale_window_type: Optional[ScaleWindowType] = None
    missing_strategy: MissingStrategy = "ignore"
    categorical_ordinal_values_order: Optional[Dict[int | str, List[str]]] = {}
    categorical_ordinal_calculation_type: Optional[CategoricalOrdinalCalcType] = "kaufman"
    feature_weights: Optional[Union[str, Dict[int | str, float]]] = {}
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

    @field_validator("categorical_ordinal_calculation_type")
    def check_categorical_ordinal_calculation_type(cls, v, values):
        if v is not None and v not in ("kaufman", "podani"):
            raise ValueError(f"Invalid categorical ordinal calculation type: {v}")
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

    @field_validator("k_neighbours")
    def check_k_neighbours(cls, v):
        if v is not None and not isinstance(v, int):
            raise ValueError(f"k_neighbours must be an int or None, got {type(v).__name__}")
        return v

    #TODO move it to better place
    # @field_validator("conditional_distances")
    # def check_conditional_distances(cls, v):
    #     if v not in (True, False):
    #         raise ValueError(f"conditional_distances must be True or False, got {v}")
    #     # Optional: validate that conditional distances make sense
    #     if v:
    #         n_feats = len(cls.__fields__['feature_types'])
    #         n_cat = sum(1 for t in cls.__fields__['feature_types'] if t.startswith("categorical"))
    #         if n_cat == 0 or n_feats == n_cat:
    #             raise ValueError("For conditional distances both categorical and numerical features must be present")
    #     return v

    @field_validator("conditional_distances_threshold_coeff")
    def check_threshold_coeff(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"conditional_distances_threshold_coeff must be at least 1, got {v}"
            )
        return v
