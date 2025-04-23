import math
import pandas as pd
from typing import Any, Sequence, Optional

def is_missing(value: Any) -> bool:
    """
    Returns True if the value is considered missing.
    
    Args:
        value: The value to check.
        
    Returns:
        bool: True if the value is missing, False otherwise.
    """

    if value is None:
        return True
    
    if isinstance(value, float) and math.isnan(value):
        return True
    
    if hasattr(pd, "isna") and pd.isna(value):
        return True
    
    return False

def first_not_missing(sequence: Sequence) -> Optional[Any]:
    """
    Returns the first non-missing value from a sequence.
    
    Args:
        sequence: A sequence of values.
        
    Returns:
        Optional[Any]: The first non-missing value, or None if all values are missing.
    """
    
    for value in sequence:
        if not is_missing(value):
            return value
    return None