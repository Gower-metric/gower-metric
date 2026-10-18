"""Gower metric C++ core (nanobind bindings)"""

from collections.abc import Mapping, Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


class CppConfigData:
    def __init__(self) -> None: ...

    @property
    def feature_types(self) -> list[str]: ...

    @feature_types.setter
    def feature_types(self, arg: Sequence[str], /) -> None: ...

    @property
    def feature_weights(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @feature_weights.setter
    def feature_weights(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def ranges(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @ranges.setter
    def ranges(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def bandwidths(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @bandwidths.setter
    def bandwidths(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def lower(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @lower.setter
    def lower(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def upper(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @upper.setter
    def upper(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def out_of_range(self) -> str: ...

    @out_of_range.setter
    def out_of_range(self, arg: str, /) -> None: ...

    @property
    def missing_strategy(self) -> str: ...

    @missing_strategy.setter
    def missing_strategy(self, arg: str, /) -> None: ...

    @property
    def categorical_ordinal_calculation_type(self) -> str: ...

    @categorical_ordinal_calculation_type.setter
    def categorical_ordinal_calculation_type(self, arg: str, /) -> None: ...

    @property
    def conditional_distances(self) -> bool: ...

    @conditional_distances.setter
    def conditional_distances(self, arg: bool, /) -> None: ...

    @property
    def conditional_distances_threshold_coeff(self) -> int: ...

    @conditional_distances_threshold_coeff.setter
    def conditional_distances_threshold_coeff(self, arg: int, /) -> None: ...

    @property
    def handle_unseen_binary_asymmetric(self) -> str: ...

    @handle_unseen_binary_asymmetric.setter
    def handle_unseen_binary_asymmetric(self, arg: str, /) -> None: ...

    @property
    def handle_unseen_binary_symmetric(self) -> str: ...

    @handle_unseen_binary_symmetric.setter
    def handle_unseen_binary_symmetric(self, arg: str, /) -> None: ...

    @property
    def handle_unseen_categorical_nominal(self) -> str: ...

    @handle_unseen_categorical_nominal.setter
    def handle_unseen_categorical_nominal(self, arg: str, /) -> None: ...

    @property
    def handle_unseen_categorical_ordinal(self) -> str: ...

    @handle_unseen_categorical_ordinal.setter
    def handle_unseen_categorical_ordinal(self, arg: str, /) -> None: ...

    @property
    def ordinal_counts(self) -> dict[int, list[float]]: ...

    @ordinal_counts.setter
    def ordinal_counts(self, arg: Mapping[int, Sequence[float]], /) -> None: ...

    @property
    def excluded_features(self) -> list[int]: ...

    @excluded_features.setter
    def excluded_features(self, arg: Sequence[int], /) -> None: ...

    @property
    def category_values(self) -> dict[int, list[float]]: ...

    @category_values.setter
    def category_values(self, arg: Mapping[int, Sequence[float]], /) -> None: ...

class CppConfig:
    def __init__(self) -> None: ...

    def configure_arguments(self, data: CppConfigData) -> None:
        """Populate the config from a ConfigData."""

    @property
    def n_features(self) -> int: ...

    @property
    def feature_weights(self) -> list[float]: ...

    @property
    def ranges(self) -> list[float]: ...

    @property
    def bandwidths(self) -> list[float]: ...

    def calculate_distance(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], y: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> float:
        """Gower distance between two 1-D feature vectors."""

    @overload
    def calculate_matrix(self, data: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None: ...

    @overload
    def calculate_matrix(self, data: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None: ...

    @overload
    def calculate_matrix(self, data: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None:
        """
        Fill a pre-allocated (n, n) matrix with pairwise Gower distances. `out` may be any supported float width; narrower than the model's own precision rounds each cell on the way out.
        """

    @overload
    def calculate_matrix_xy(self, x_data: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], y_data: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None: ...

    @overload
    def calculate_matrix_xy(self, x_data: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], y_data: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None: ...

    @overload
    def calculate_matrix_xy(self, x_data: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], y_data: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None:
        """Fill a pre-allocated (n_x, n_y) matrix with cross pairwise Gower."""

class CppConfigDataF:
    def __init__(self) -> None: ...

    @property
    def feature_types(self) -> list[str]: ...

    @feature_types.setter
    def feature_types(self, arg: Sequence[str], /) -> None: ...

    @property
    def feature_weights(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @feature_weights.setter
    def feature_weights(self, arg: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def ranges(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @ranges.setter
    def ranges(self, arg: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def bandwidths(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @bandwidths.setter
    def bandwidths(self, arg: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def lower(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @lower.setter
    def lower(self, arg: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def upper(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @upper.setter
    def upper(self, arg: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def out_of_range(self) -> str: ...

    @out_of_range.setter
    def out_of_range(self, arg: str, /) -> None: ...

    @property
    def missing_strategy(self) -> str: ...

    @missing_strategy.setter
    def missing_strategy(self, arg: str, /) -> None: ...

    @property
    def categorical_ordinal_calculation_type(self) -> str: ...

    @categorical_ordinal_calculation_type.setter
    def categorical_ordinal_calculation_type(self, arg: str, /) -> None: ...

    @property
    def conditional_distances(self) -> bool: ...

    @conditional_distances.setter
    def conditional_distances(self, arg: bool, /) -> None: ...

    @property
    def conditional_distances_threshold_coeff(self) -> int: ...

    @conditional_distances_threshold_coeff.setter
    def conditional_distances_threshold_coeff(self, arg: int, /) -> None: ...

    @property
    def handle_unseen_binary_asymmetric(self) -> str: ...

    @handle_unseen_binary_asymmetric.setter
    def handle_unseen_binary_asymmetric(self, arg: str, /) -> None: ...

    @property
    def handle_unseen_binary_symmetric(self) -> str: ...

    @handle_unseen_binary_symmetric.setter
    def handle_unseen_binary_symmetric(self, arg: str, /) -> None: ...

    @property
    def handle_unseen_categorical_nominal(self) -> str: ...

    @handle_unseen_categorical_nominal.setter
    def handle_unseen_categorical_nominal(self, arg: str, /) -> None: ...

    @property
    def handle_unseen_categorical_ordinal(self) -> str: ...

    @handle_unseen_categorical_ordinal.setter
    def handle_unseen_categorical_ordinal(self, arg: str, /) -> None: ...

    @property
    def ordinal_counts(self) -> dict[int, list[float]]: ...

    @ordinal_counts.setter
    def ordinal_counts(self, arg: Mapping[int, Sequence[float]], /) -> None: ...

    @property
    def excluded_features(self) -> list[int]: ...

    @excluded_features.setter
    def excluded_features(self, arg: Sequence[int], /) -> None: ...

    @property
    def category_values(self) -> dict[int, list[float]]: ...

    @category_values.setter
    def category_values(self, arg: Mapping[int, Sequence[float]], /) -> None: ...

class CppConfigF:
    def __init__(self) -> None: ...

    def configure_arguments(self, data: CppConfigDataF) -> None:
        """Populate the config from a ConfigData."""

    @property
    def n_features(self) -> int: ...

    @property
    def feature_weights(self) -> list[float]: ...

    @property
    def ranges(self) -> list[float]: ...

    @property
    def bandwidths(self) -> list[float]: ...

    def calculate_distance(self, x: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], y: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> float:
        """Gower distance between two 1-D feature vectors."""

    @overload
    def calculate_matrix(self, data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None: ...

    @overload
    def calculate_matrix(self, data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None: ...

    @overload
    def calculate_matrix(self, data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None:
        """
        Fill a pre-allocated (n, n) matrix with pairwise Gower distances. `out` may be any supported float width; narrower than the model's own precision rounds each cell on the way out.
        """

    @overload
    def calculate_matrix_xy(self, x_data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], y_data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None: ...

    @overload
    def calculate_matrix_xy(self, x_data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], y_data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None: ...

    @overload
    def calculate_matrix_xy(self, x_data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], y_data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None:
        """Fill a pre-allocated (n_x, n_y) matrix with cross pairwise Gower."""

class CppConfigDataH:
    def __init__(self) -> None: ...

    @property
    def feature_types(self) -> list[str]: ...

    @feature_types.setter
    def feature_types(self, arg: Sequence[str], /) -> None: ...

    @property
    def feature_weights(self) -> Annotated[NDArray[numpy.float16], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @feature_weights.setter
    def feature_weights(self, arg: Annotated[NDArray[numpy.float16], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def ranges(self) -> Annotated[NDArray[numpy.float16], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @ranges.setter
    def ranges(self, arg: Annotated[NDArray[numpy.float16], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def bandwidths(self) -> Annotated[NDArray[numpy.float16], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @bandwidths.setter
    def bandwidths(self, arg: Annotated[NDArray[numpy.float16], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def lower(self) -> Annotated[NDArray[numpy.float16], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @lower.setter
    def lower(self, arg: Annotated[NDArray[numpy.float16], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def upper(self) -> Annotated[NDArray[numpy.float16], dict(shape=(None,), order='C', device='cpu', writable=False)]: ...

    @upper.setter
    def upper(self, arg: Annotated[NDArray[numpy.float16], dict(shape=(None,), order='C', device='cpu', writable=False)], /) -> None: ...

    @property
    def out_of_range(self) -> str: ...

    @out_of_range.setter
    def out_of_range(self, arg: str, /) -> None: ...

    @property
    def missing_strategy(self) -> str: ...

    @missing_strategy.setter
    def missing_strategy(self, arg: str, /) -> None: ...

    @property
    def categorical_ordinal_calculation_type(self) -> str: ...

    @categorical_ordinal_calculation_type.setter
    def categorical_ordinal_calculation_type(self, arg: str, /) -> None: ...

    @property
    def conditional_distances(self) -> bool: ...

    @conditional_distances.setter
    def conditional_distances(self, arg: bool, /) -> None: ...

    @property
    def conditional_distances_threshold_coeff(self) -> int: ...

    @conditional_distances_threshold_coeff.setter
    def conditional_distances_threshold_coeff(self, arg: int, /) -> None: ...

    @property
    def handle_unseen_binary_asymmetric(self) -> str: ...

    @handle_unseen_binary_asymmetric.setter
    def handle_unseen_binary_asymmetric(self, arg: str, /) -> None: ...

    @property
    def handle_unseen_binary_symmetric(self) -> str: ...

    @handle_unseen_binary_symmetric.setter
    def handle_unseen_binary_symmetric(self, arg: str, /) -> None: ...

    @property
    def handle_unseen_categorical_nominal(self) -> str: ...

    @handle_unseen_categorical_nominal.setter
    def handle_unseen_categorical_nominal(self, arg: str, /) -> None: ...

    @property
    def handle_unseen_categorical_ordinal(self) -> str: ...

    @handle_unseen_categorical_ordinal.setter
    def handle_unseen_categorical_ordinal(self, arg: str, /) -> None: ...

    @property
    def ordinal_counts(self) -> dict[int, list[float]]: ...

    @ordinal_counts.setter
    def ordinal_counts(self, arg: Mapping[int, Sequence[float]], /) -> None: ...

    @property
    def excluded_features(self) -> list[int]: ...

    @excluded_features.setter
    def excluded_features(self, arg: Sequence[int], /) -> None: ...

    @property
    def category_values(self) -> dict[int, list[float]]: ...

    @category_values.setter
    def category_values(self, arg: Mapping[int, Sequence[float]], /) -> None: ...

class CppConfigH:
    def __init__(self) -> None: ...

    def configure_arguments(self, data: CppConfigDataH) -> None:
        """Populate the config from a ConfigData."""

    @property
    def n_features(self) -> int: ...

    @property
    def feature_weights(self) -> list[float]: ...

    @property
    def ranges(self) -> list[float]: ...

    @property
    def bandwidths(self) -> list[float]: ...

    def calculate_distance(self, x: Annotated[NDArray[numpy.float16], dict(shape=(None,), order='C', device='cpu', writable=False)], y: Annotated[NDArray[numpy.float16], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> float:
        """Gower distance between two 1-D feature vectors."""

    @overload
    def calculate_matrix(self, data: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None: ...

    @overload
    def calculate_matrix(self, data: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None: ...

    @overload
    def calculate_matrix(self, data: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None:
        """
        Fill a pre-allocated (n, n) matrix with pairwise Gower distances. `out` may be any supported float width; narrower than the model's own precision rounds each cell on the way out.
        """

    @overload
    def calculate_matrix_xy(self, x_data: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu', writable=False)], y_data: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None: ...

    @overload
    def calculate_matrix_xy(self, x_data: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu', writable=False)], y_data: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None: ...

    @overload
    def calculate_matrix_xy(self, x_data: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu', writable=False)], y_data: Annotated[NDArray[numpy.float16], dict(shape=(None, None), order='C', device='cpu', writable=False)], out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], similarity: bool = False) -> None:
        """Fill a pre-allocated (n_x, n_y) matrix with cross pairwise Gower."""
