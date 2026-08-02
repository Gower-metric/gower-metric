# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

"""Shared base class for the precision tests."""

from typing import cast

import numpy as np
import pytest

from tests.conftest import NUMPY_NUMERIC_TYPES


class BaseTest:
    """Base test class that provides float dtype parameterization to all subclasses."""

    dtype: type[np.floating]
    """The float precision under test, refreshed per parameterisation."""

    @pytest.fixture(autouse=True, params=NUMPY_NUMERIC_TYPES)
    def setup_dtype(self, request: pytest.FixtureRequest) -> None:
        """Assign numpy float dtype to self.dtype for all tests in subclasses."""
        self.dtype = cast("type[np.floating]", request.param)
