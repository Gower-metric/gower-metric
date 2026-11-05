# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

from .config import Config
from .exceptions import IllegalStateError
from .metric import Gower

__all__ = ["Config", "Gower", "IllegalStateError"]

__doc__ = """
Gower-metric: A Python package for computing Gower distance, similarity and matrices.
"""
