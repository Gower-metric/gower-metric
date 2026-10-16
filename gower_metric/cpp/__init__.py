# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

"""Native (C++/nanobind) core for the gower metric."""

from typing import TYPE_CHECKING

from gower_metric.cpp.cpp_engine import (
    CppConfig,
    CppConfigData,
    CppConfigDataF,
    CppConfigF,
)

if TYPE_CHECKING:
    from gower_metric.cpp.cpp_engine import CppConfigDataH, CppConfigH

    HAS_FLOAT16 = True
else:
    try:
        from gower_metric.cpp.cpp_engine import CppConfigDataH, CppConfigH

        HAS_FLOAT16 = True
    except ImportError:  # pragma: no cover
        CppConfigH = None
        CppConfigDataH = None
        HAS_FLOAT16 = False

__all__ = [
    "HAS_FLOAT16",
    "CppConfig",
    "CppConfigData",
    "CppConfigDataF",
    "CppConfigDataH",
    "CppConfigF",
    "CppConfigH",
]
