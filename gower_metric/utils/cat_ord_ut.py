from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def map_ordered_values(
    ordered_values: Sequence[Any] | np.ndarray,
) -> tuple[dict[Any, int], int | None, int | None]:
    """Map consecutive integers to passed ordered values.

    Args:
        ordered_values (Sequence[Any] | np.ndarray): A defined sequence of categorical values.

    Returns:
        tuple[dict[Any, int], int | None, int | None]:
            - ranks_mapping: A dictionary mapping each unique value to its rank.
            - min_rank: The minimum rank (or None if no categories).
            - max_rank: The maximum rank (or None if no categories).

    """
    ranks_mapping = {value: rank for rank, value in enumerate(ordered_values)}
    if len(ordered_values) == 0:
        return ranks_mapping, None, None
    min_rank: int | None = 0
    max_rank: int | None = len(ordered_values) - 1

    return ranks_mapping, min_rank, max_rank


def get_cardinalities_mapping(
    column: Sequence[Any] | np.ndarray,
) -> tuple[dict[Any, int], list[int]]:
    """Count occurrences of each category value in an ordinal column.

    Args:
        column (Sequence[Any] | np.ndarray): A sequence of ordinal values (may include NaN).
            NaN values are ignored in counting.

    Returns:
        tuple[dict[Any, int], list[int]]:
            - counts_map: Mapping from each unique category value (excluding NaN) to its count.
            - counts_list: List of counts corresponding to each category value, ordered by sorted category values.

    """
    cleaned = (
        column[~pd.isna(column)]
        if isinstance(column, np.ndarray)
        else [v for v in column if not pd.isna(v)]
    )
    counts_map: dict[Any, int] = Counter(cleaned)

    unique_vals = sorted(counts_map.keys())
    counts_list = [counts_map[val] for val in unique_vals]

    return counts_map, counts_list
