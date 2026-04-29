from .core.config import Config
from .core.exceptions import IllegalStateError
from .core.metric import Gower

__all__ = ["Config", "Gower", "IllegalStateError"]

__doc__ = """
Gower metric for mixed data types.
Gower metric is a distance metric that can handle mixed data types,
including numerical, categorical, binary, and ordinal data.
It is particularly useful in scenarios where datasets contain a combination
of different types of variables.
"""
