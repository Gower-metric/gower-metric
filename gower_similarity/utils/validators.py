from typing import Dict, Any

ALLOWED_FEATURE_TYPES = {
    "numeric",
    "categorical_nominal",
    "categorical_ordinal",
    "binary_asymmetric",
    "binary_symmetric",
    "ratio_scale_interval",
}
ALLOWED_SCALE_METHODS = {"range", "iqr"}
ALLOWED_MISSING_STRATEGIES = {"ignore", "max_dist", "raise_error"}


def validate_feature_types(feature_types: Dict[Any, str]) -> None:
    """
    Validate the feature types dictionary.
    
    Args:
        feature_types (dict): A dictionary mapping column names to their feature types.
        
    Raises:
        ValueError: If the feature types are not valid.
    """
    if not isinstance(feature_types, dict) or not feature_types:
        raise ValueError(
            "feature_types must be a non-empty dict mapping columns to one of "
            f"{sorted(ALLOWED_FEATURE_TYPES)}")
    for k, v in feature_types.items():
        if v not in ALLOWED_FEATURE_TYPES:
            raise ValueError(f"Unknown feature type '{v}' for column {k!r}. "
                             f"Allowed types: {sorted(ALLOWED_FEATURE_TYPES)}")


def validate_scale_method(scale: str) -> None:
    """
    Validate the scale method.

    Args:
        scale (str): The scale method to validate.

    Raises:
        ValueError: If the scale method is not valid.
    """
    if scale.lower() not in ALLOWED_SCALE_METHODS:
        raise ValueError(
            f"scale must be one of {sorted(ALLOWED_SCALE_METHODS)}, got '{scale}'"
        )


def validate_missing_strategy(missing_strategy: str) -> None:
    """
    Validate the missing strategy.

    Args:
        missing_strategy (str): The missing strategy to validate.
    
    Raises:
        ValueError: If the missing strategy is not valid.
    """
    if missing_strategy.lower() not in ALLOWED_MISSING_STRATEGIES:
        raise ValueError(
            f"missing_strategy must be one of {sorted(ALLOWED_MISSING_STRATEGIES)}, "
            f"got '{missing_strategy}'")
