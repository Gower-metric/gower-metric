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
    transformed = gower.transform(data)

    # features are sorted alphabetically, so A -> 0.0 and B -> 1.0

    np.testing.assert_array_equal(
        transformed["feature1"].to_numpy(),
        [0.0, 1.0, 0.0, 1.0],
    )
    np.testing.assert_array_equal(
        transformed["feature2"].to_numpy(),
        [0.0, 0.0, 1.0, 1.0],
    )


def test_ordinal_consistency() -> None:
    """Test consistency of ordinal features in train/test split.

    Order is defined in config: ['low', 'medium', 'high'].
    Fit on ['low', 'medium'].
    Transform on ['high'] -> should work (2.0).
    Transform on ['extra'] -> should be NaN (unseen/invalid).
    """
    X_train = pd.DataFrame({"ord": ["low", "medium"]})
    X_test = pd.DataFrame({"ord": ["high", "extra"]})

    cfg = Config(
        feature_types={"ord": "categorical_ordinal"},
        categorical_ordinal_values_order={"ord": ["low", "medium", "high"]},
    )
    gower = Gower(cfg)
    gower.fit(X_train)

    t_train = gower.transform(X_train)
    res_train = t_train["ord"].to_numpy()

    np.testing.assert_array_equal(res_train, [0.0, 1.0])

    # high->2, extra->NaN
    t_test = gower.transform(X_test)
    res_test = t_test["ord"].to_numpy()

    np.testing.assert_array_equal(res_test, [2.0, np.nan])


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
def test_binary_unseen_value_degenerate_fit(bin_type) -> None:
    """Test that transform maps to NaN if fit saw only 1 value (degenerate)."""
    data_fit = pd.DataFrame({"feature1": ["A"]})
    data_transform = pd.DataFrame({"feature1": ["B"]})

    if bin_type == "binary_asymmetric":
        cfg = Config(
            feature_types={"feature1": bin_type},
            handle_unseen_binary_asymmetric="missing",
        )
    else:
        cfg = Config(
            feature_types={"feature1": bin_type},
            handle_unseen_binary_symmetric="missing",
        )

    gower = Gower(cfg)
    gower.fit(data_fit)

    transformed = gower.transform(data_transform)

    # 'A' -> {A: 0.0}. 'B' is unseen, maps to NaN
    expected = np.array([np.nan])

    res = transformed["feature1"].to_numpy()

    np.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize("bin_type", ["binary_symmetric", "binary_asymmetric"])
def test_binary_unseen_value_complete_fit(bin_type) -> None:
    """Test that transform raises ValueError if fit saw 2 values (complete)."""
    data_fit = pd.DataFrame({"feature1": ["A", "B"]})
    data_transform = pd.DataFrame({"feature1": ["C"]})

    if bin_type == "binary_asymmetric":
        cfg = Config(
            feature_types={"feature1": bin_type},
            handle_unseen_binary_asymmetric="error",
        )
    else:
        cfg = Config(
            feature_types={"feature1": bin_type},
            handle_unseen_binary_symmetric="error",
        )

    gower = Gower(cfg)
    gower.fit(data_fit)

    with pytest.raises(ValueError, match=r"has 3 unique values total"):
        gower.transform(data_transform)
