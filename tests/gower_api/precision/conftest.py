"""Conftest for precision tests: parametrizes every test by numpy float dtype."""

import pytest

from tests.conftest import NUMPY_NUMERIC_TYPES


class BaseTest:
    """Base test class that provides float dtype parameterization to all subclasses."""

    @pytest.fixture(autouse=True, params=NUMPY_NUMERIC_TYPES)
    def setup_dtype(self, request) -> None:
        """Assign numpy float dtype to self.dtype for all tests in subclasses."""
        self.dtype = request.param
