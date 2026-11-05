# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

"""The float16 native classes are optional; without them float16 is reported missing."""

import importlib
import sys
import types
from collections.abc import Iterator

import pytest

import gower_metric.cpp as cpp_package


@pytest.fixture
def _restore_cpp_package() -> Iterator[None]:
    """Reload the real package after a test has swapped the engine out."""
    original = sys.modules["gower_metric.cpp.cpp_engine"]
    try:
        yield
    finally:
        sys.modules["gower_metric.cpp.cpp_engine"] = original
        importlib.reload(cpp_package)


@pytest.mark.usefixtures("_restore_cpp_package")
def test_missing_half_precision_is_reported_as_unavailable() -> None:
    engine = sys.modules["gower_metric.cpp.cpp_engine"]
    stub = types.ModuleType("gower_metric.cpp.cpp_engine")
    for name in ("CppConfig", "CppConfigData", "CppConfigDataF", "CppConfigF"):
        setattr(stub, name, getattr(engine, name))
    sys.modules["gower_metric.cpp.cpp_engine"] = stub

    reloaded = importlib.reload(cpp_package)

    assert reloaded.HAS_FLOAT16 is False
    assert reloaded.CppConfigH is None
    assert reloaded.CppConfigDataH is None
