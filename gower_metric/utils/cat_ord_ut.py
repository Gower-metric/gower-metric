import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np


def map_ordered_values(
    ordered_values: Sequence[Any] | np.ndarray,
) -> tuple[dict[Any, int], int | None, int | None]:
    """
    Map consequtive integers to passed ordered values.

    Args:
        ordered_values: A defined sequence of categorical values.

    Returns:
        A tuple containing:
            - ranks_mapping: A dictionary mapping each unique value to its rank.
            - min_rank: The minimum rank (or None if no categories).
            - max_rank: The maximum rank (or None if no categories).
    """
    ranks_mapping = {value: rank for rank, value in enumerate(ordered_values)}
    min_rank = 0
    max_rank = len(ordered_values) - 1

    return ranks_mapping, min_rank, max_rank


def get_cardinalities_mapping(
    column: Sequence[Any] | np.ndarray,
) -> tuple[dict[Any, int], list[int]]:
    """
    Count occurrences of each category value in an ordinal column.

    Args:
        column : Sequence[Any]
            A sequence of ordinal values (may include NaN).
            NaN values are ignored in counting.

    Returns:
        counts_map : Dict[Any, int]
            Mapping from each unique category value (excluding NaN) to its count.
        counts_list : List[int]
            List of counts corresponding to each category value,
            ordered by sorted category values.
    """
    cleaned = [v for v in column if not (isinstance(v, float) and math.isnan(v))]
    counts_map: dict[Any, int] = Counter(cleaned)

    unique_vals = sorted(counts_map.keys())
    counts_list = [counts_map[val] for val in unique_vals]

    return counts_map, counts_list


def collect_ordinal_cardinalities(data: np.ndarray) -> list[np.ndarray]:
    """
    Process a 2D array of ordinal columns to get counts per level for each column.

    Args:
        data : np.ndarray
            Two-dimensional array with shape (n_samples, n_ordinal_columns).
            Each column may contain NaN and ordinal categorical values.

    Returns:
        ordinals_cardinality : List[np.ndarray]
            A list where each element is a 1D NumPy array of integer counts.
            counts[i] is the number of occurrences of the i-th sorted category in that column.
    """
    ordinals_cardinality: list[np.ndarray] = []
    for i in range(data.shape[1]):
        column = data[:, i]
        _, counts_list = get_cardinalities_mapping(column)
        ordinals_cardinality.append(np.array(counts_list, dtype=int))
    return ordinals_cardinality
