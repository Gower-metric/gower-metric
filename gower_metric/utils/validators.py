from typing import Any

ALLOWED_FEATURE_TYPES = {
    "numeric",
    "categorical_nominal",
    "categorical_ordinal",
    "binary_asymmetric",
    "binary_symmetric",
    "ratio_scale_interval",
}
ALLOWED_SCALE_METHODS = {"range", "iqr"}
ALLOWED_SCALE_WINDOWS = {None, "kde", "kNN"}
ALLOWED_SCALE_WINDOWS_TYPES = {None, "silverman"}
ALLOWED_MISSING_STRATEGIES = {"ignore", "max_dist", "raise_error"}
ALLOWED_CATEGORICAL_ORDINAL_CALCULATION_TYPES = {"kaufman", "podani"}
ALLOWED_WEIGHTS_TYPES = {None, "uniform"}
ALLOWED_K_NEIGHBOURS_TYPES = {None, int}
ALLOWED_CONDITIONAL_DISTANCES = {False, True}
ALLOWED_CONDITIONAL_DISTANCES_THRESHOLD_COEFF_MIN_VALUE = 1


def validate_feature_types(feature_types: dict[Any, str]) -> None:
    """
    Validate the feature types dictionary.

    Args:
        feature_types (dict[Any, str]): A dictionary mapping column names to their feature types.

    Raises:
        ValueError: If the feature types are not valid.
    """
    if not isinstance(feature_types, dict) or not feature_types:
        raise ValueError(
            "feature_types must be a non-empty dict mapping columns to one of "
            f"{(ALLOWED_FEATURE_TYPES)}"
        )
    for k, v in feature_types.items():
        if v not in ALLOWED_FEATURE_TYPES:
            raise ValueError(
                f"Unknown feature type '{v}' for column {k!r}. "
                f"Allowed types: {(ALLOWED_FEATURE_TYPES)}"
            )


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
            f"scale must be one of {(ALLOWED_SCALE_METHODS)}, got '{scale}'"
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
            f"missing_strategy must be one of {(ALLOWED_MISSING_STRATEGIES)}, "
            f"got '{missing_strategy}'"
        )


def validate_categorical_ordinal_values_order(
    categorical_ordinal_values_order: dict[int | str, list[str]],
    feature_types: dict[int | str, str],
) -> None:
    """
    Validate whether all defined categorical ordinal columns have specified the order of their values.

    Args:
        categorical_ordinal_values_order (dict[int | str, list[str]]): categorical ordinal values order to validate.
        feature_types (dict[int | str, str]): A dictionary mapping column names to their feature types.

    Raises:
        ValueError: If the categorical ordinal values order is not valid.
    """
    categorical_ordinal_columns = {
        k for k, v in feature_types.items() if v == "categorical_ordinal"
    }
    defined_values_order_columns = set(categorical_ordinal_values_order.keys())

    if defined_values_order_columns < categorical_ordinal_columns:
        raise ValueError(
            f"Categorical ordinal values order must be defined for the following columns: "
            f"{categorical_ordinal_columns}, "
            f"got {defined_values_order_columns}"
        )


def validate_categorical_ordinal_calculation_type(calculation_type: str) -> None:
    """
    Validate the calculation type for categorical nominal features.

    Args:
        calculation_type (str): The calculation type to validate.

    Raises:
        ValueError: If the calculation type is not valid.
    """
    if calculation_type.lower() not in ALLOWED_CATEGORICAL_ORDINAL_CALCULATION_TYPES:
        raise ValueError(
            f"calculation_type must be one of "
            f"{(ALLOWED_CATEGORICAL_ORDINAL_CALCULATION_TYPES)}, "
            f"got '{calculation_type}'"
        )


def validate_scale_window_and_type(
    scale_window: str | None, scale_window_type: str | None
) -> None:
    """
    Validate the scale window and it's type at the same time.

    Args:
        scale_window (Optional[str]): The scale window to validate.
        scale_window_type (Optional[str]): The scale window type to validate.

    Raises:
        ValueError: If the scale window is not valid.
    """
    if scale_window not in ALLOWED_SCALE_WINDOWS:
        raise ValueError(
            f"scale_window must be one of {(ALLOWED_SCALE_WINDOWS)}, got {scale_window!r}"
        )

    if scale_window is None:
        if scale_window_type is not None:
            raise ValueError(
                f"scale_window_type must be None when scale_window is None, got {scale_window_type!r}"
            )
        return

    if scale_window == "kde" and scale_window_type not in ALLOWED_SCALE_WINDOWS_TYPES:
        raise ValueError(
            f"scale_window_type must be one of {(ALLOWED_SCALE_WINDOWS_TYPES)}, "
            f"got {scale_window_type!r} when scale_window='kde'"
        )


def validate_weights_type(weights: str | dict) -> None:
    """
    Validate the weights type.

    Args:
        weights (str | dict): The weights to validate.

    Raises:
        ValueError: If the weights type is not valid.
    """
    if weights is None or isinstance(weights, str):
        if weights not in ALLOWED_WEIGHTS_TYPES:
            raise ValueError(
                f"weights must be one of {(ALLOWED_WEIGHTS_TYPES)}, got {weights!r}"
            )
    elif not isinstance(weights, dict):
        raise ValueError(
            "weights must be None, a string, or a dictionary mapping feature "
            "indices to weights, got {type(weights).__name__}"
        )


def validate_k_neighbours(k_neighbours: int | None) -> None:
    """
    Validate the k-neighbours type.

    Args:
        k_neighbours (int | None): The k-neighbours to validate.

    Raises:
        ValueError: If the k-neighbours type is not valid.
    """
    if k_neighbours is None or isinstance(k_neighbours, str):
        if k_neighbours not in ALLOWED_K_NEIGHBOURS_TYPES:
            raise ValueError(
                f"k_neighbours must be one of {(ALLOWED_K_NEIGHBOURS_TYPES)}, "
                f"got {k_neighbours!r}"
            )
    elif not isinstance(k_neighbours, int):
        raise ValueError(
            "k_neighbours must be None, a string, or an integer, got "
            f"{type(k_neighbours).__name__}"
        )


def validate_conditional_distances(conditional_distances: bool) -> None:
    """
    Validate the conditional distances flag.

    Args:
        conditional_distances (bool): Flag to validate

    Raises:
        ValueError: If conditional_distances flag different from bool
    """
    if conditional_distances not in ALLOWED_CONDITIONAL_DISTANCES:
        raise ValueError(
            f"Conditional_distances flag must be one of {ALLOWED_CONDITIONAL_DISTANCES}, "
            f"got {conditional_distances}"
        )


def validate_conditional_distances_threshold_coeff(
    conditional_distances_threshold_coeff: int,
) -> None:
    """
    Validate the conditional distances threshold coefficient.

    Args:
        conditional_distances_threshold_coeff (int): Value of the threshold coefficient

    Raises:
        ValueError: If conditional_distances_threshold_coeff not an int or lower than 1.
    """
    if (
        not isinstance(conditional_distances_threshold_coeff, int)
        or conditional_distances_threshold_coeff
        < ALLOWED_CONDITIONAL_DISTANCES_THRESHOLD_COEFF_MIN_VALUE
    ):
        raise ValueError(
            f"Conditional_distances_threshold_coeff must be of type `int` "
            f"at least equal to {ALLOWED_CONDITIONAL_DISTANCES_THRESHOLD_COEFF_MIN_VALUE}, "
            f"got {conditional_distances_threshold_coeff}"
        )


def validate_feature_types_for_conditional_distances(n_feats: int, p_cat: int) -> None:
    """
    Validate the data passed to use with the conditional ditances.

    Args:
        n_feats (int): Number of passed features
        p_cat (int): Number of categorical features

    Raises:
        ValueError: If there are either no categorical or no numerical features passed.
    """
    if p_cat in (0, n_feats):
        raise ValueError(
            "For computing conditional distances both type of data: categorical and numerical need to be provided."
        )
