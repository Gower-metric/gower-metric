from typing import cast

import numpy as np
import pandas as pd
import pytest

from gower_metric import Config, Gower


@pytest.mark.parametrize("bin_type", ["binary_symmetric", "binary_asymmetric"])
def test_binary_ab_mapping(bin_type) -> None:
    """Test that arbitrary binary values (e.g., 'A', 'B') are correctly mapped."""
    data = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "B"],
            "feature2": ["A", "A", "B", "B"],
        },
    )
    feature_types: dict[int | str, str] = {
        "feature1": bin_type,
        "feature2": bin_type,
    }
    cfg = Config(feature_types=feature_types)
    gower = Gower(cfg)

    gower.fit(data)
    transformed = cast("pd.DataFrame", gower.transform(data))

    # feature are sorted alphabetically, so A -> 0.0 and B -> 1.0

    np.testing.assert_array_equal(transformed["feature1"].values, [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(transformed["feature2"].values, [0.0, 0.0, 1.0, 1.0])


@pytest.mark.parametrize("bin_type", ["binary_symmetric", "binary_asymmetric"])
def test_binary_too_many_values(bin_type) -> None:
    """Test that fit raises ValueError if a binary column has > 2 unique values."""
    data = pd.DataFrame(
        {
            "feature1": ["A", "B", "C", "A"],
        },
    )
    feature_types: dict[int | str, str] = {
        "feature1": bin_type,
    }
    cfg = Config(feature_types=feature_types)
    gower = Gower(cfg)

    with pytest.raises(ValueError, match="more than 2 unique values"):
        gower.fit(data)


@pytest.mark.parametrize("bin_type", ["binary_symmetric", "binary_asymmetric"])
def test_binary_unseen_value_in_transform(bin_type) -> None:
    """Test that transform raises ValueError if it encounters a value not seen in fit."""
    data_fit = pd.DataFrame(
        {
            "feature1": ["A", "B"],
        },
    )
    data_transform = pd.DataFrame(
        {
            "feature1": ["A", "C"],
        },
    )
    feature_types: dict[int | str, str] = {
        "feature1": bin_type,
    }
    cfg = Config(feature_types=feature_types)
    gower = Gower(cfg)

    gower.fit(data_fit)

    with pytest.raises(ValueError, match="not found in fitted binary mapping"):
        gower.transform(data_transform)
