"""Tests for weights.py — uniform, None, dict, and invalid config."""

import numpy as np
import pytest

from gower_metric.weights.weights import get_weights


def test_weights_none_returns_ones() -> None:
    w = get_weights(5, config=None)
    np.testing.assert_array_equal(w, np.ones(5))


def test_weights_uniform_string_returns_ones() -> None:
    w = get_weights(3, config="uniform")
    np.testing.assert_array_equal(w, np.ones(3))


def test_weights_dict_applies_values() -> None:
    w = get_weights(4, config={0: 2.0, 2: 0.5})
    assert w[0] == 2.0
    assert w[1] == 1.0
    assert w[2] == 0.5
    assert w[3] == 1.0


def test_weights_invalid_config_raises() -> None:
    with pytest.raises(ValueError, match="config must be None"):
        get_weights(3, config=42)  # type: ignore[arg-type]
