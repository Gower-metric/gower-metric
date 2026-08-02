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
else:
    try:
        from gower_metric.cpp.cpp_engine import CppConfigDataH, CppConfigH
    except ImportError:
        CppConfigH = CppConfigF
        CppConfigDataH = CppConfigDataF

__all__ = [
    "CppConfig",
    "CppConfigData",
    "CppConfigDataF",
    "CppConfigDataH",
    "CppConfigF",
    "CppConfigH",
]
