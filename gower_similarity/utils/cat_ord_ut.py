from typing import Tuple, Dict, Sequence, Any, Optional

def get_ranks_mapping(
    column: Sequence[Any]
) -> Tuple[Dict[Any, int], Optional[int], Optional[int]]:
    """
    Get ranks mapping for a categorical column.

    Args:
        column: A sequence of categorical values.

    Returns:
        A tuple containing:
            - ranks_mapping: A dictionary mapping each unique value to its rank.
            - min_rank: The minimum rank (or None if no categories).
            - max_rank: The maximum rank (or None if no categories).
    """
    unique_values = list(dict.fromkeys(column))
    if not unique_values:
        return {}, None, None

    ranks_mapping = {value: rank for rank, value in enumerate(unique_values)}
    min_rank = 0
    max_rank = len(unique_values) - 1

    return ranks_mapping, min_rank, max_rank