from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def map_ordered_values(
    ordered_values: Sequence[Any] | np.ndarray,
    data_type: type[np.floating] = np.float32,
) -> tuple[dict[Any, int], int | None, int | None]:
    """Map consecutive integer ranks to passed ordered values.

    The returned mapping is lookup-ready for both raw user values (e.g. strings)
    and post-``Gower.transform`` numeric ranks — callers can feed either form
    into ``categorical_ordinal_component`` without post-processing the map.

    Args:
        ordered_values (Sequence[Any] | np.ndarray): A defined sequence of
            categorical values in raw form.
        data_type (type[np.floating]): Floating dtype used by ``OrdinalEncoder``
            in the Gower pipeline. Encoded-form keys are inserted with this
            dtype so lookups with transformed columns succeed without casting.

    Returns:
        tuple[dict[Any, int], int | None, int | None]:
            - ranks_mapping: ``{raw_value: rank, data_type(rank): rank, int(rank): rank}``
              for each position. Values are integer ranks.
            - min_rank: 0, or None if ``ordered_values`` is empty.
            - max_rank: ``len(ordered_values) - 1``, or None if empty.

    """
    if len(ordered_values) == 0:
        return {}, None, None

    ranks_mapping: dict[Any, int] = {}
    for rank, value in enumerate(ordered_values):
        ranks_mapping[value] = rank
        ranks_mapping[data_type(rank)] = rank
        ranks_mapping[int(rank)] = rank

    return ranks_mapping, 0, len(ordered_values) - 1


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
