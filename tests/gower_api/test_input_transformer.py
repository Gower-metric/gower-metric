import numpy as np
import pandas as pd
import pytest

from gower_metric import Gower
from gower_metric.core.config import Config
from gower_metric.core.exceptions import IllegalStateError

DTYPE = np.float16


def test_transform_before_fit() -> None:
    data = np.array([[0], [1], [0]], dtype=object)

    cfg = Config(
        feature_types={0: "binary_symmetric"},
    )
    gower = Gower(cfg)

    with pytest.raises(
        IllegalStateError,
        match="Operation not allowed: model is not fitted",
    ):
        gower.transform(data)


def test_transform_with_np_array() -> None:
    data = np.array(
        [
            ["low", True, False, "car", 4.5],
            ["medium", False, True, "plane", 1.8],
            ["high", False, True, "car", 0.95],
            ["low", True, False, "train", 1.11],
        ],
        dtype=object,
    )
    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
        0: ["low", "medium", "high"],
    }

    cfg = Config(
        feature_types={
            0: "categorical_ordinal",
            1: "binary_asymmetric",
            2: "binary_symmetric",
            3: "categorical_nominal",
            4: "ratio_scale_interval",
        },
        data_type=DTYPE,
        categorical_ordinal_values_order=categorical_ordinal_values_order,
    )
    gower = Gower(cfg).fit(data)
    transformed_data = gower.transform(data)

    expected_data = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 4.5],
            [1.0, 0.0, 1.0, 1.0, 1.8],
            [2.0, 0.0, 1.0, 0.0, 0.95],
            [0.0, 1.0, 0.0, 2.0, 1.11],
        ],
        dtype=DTYPE,
    )

    np.testing.assert_array_equal(transformed_data, expected_data)


def test_transform_with_df() -> None:
    data = pd.DataFrame(
        {
            "level": ["low", "medium", "high", "low"],
            "positive": [True, False, False, True],
            "negative": [False, True, True, False],
            "type": ["car", "plane", "car", "train"],
            "price": [4.5, 1.8, 0.95, 1.11],
        },
    )

    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
        "level": ["low", "medium", "high"],
    }

    cfg = Config(
        feature_types={
            "level": "categorical_ordinal",
            "positive": "binary_asymmetric",
            "negative": "binary_symmetric",
            "type": "categorical_nominal",
            "price": "ratio_scale_interval",
        },
        data_type=DTYPE,
        categorical_ordinal_values_order=categorical_ordinal_values_order,
    )
    gower = Gower(cfg).fit(data)
    transformed_data = gower.transform(data)

    expected_data = pd.DataFrame(
        {
            "level": [0.0, 1.0, 2.0, 0.0],
            "positive": [1.0, 0.0, 0.0, 1.0],
            "negative": [0.0, 1.0, 1.0, 0.0],
            "type": [0.0, 1.0, 0.0, 2.0],
            "price": [4.5, 1.8, 0.95, 1.11],
        },
        dtype=DTYPE,
    )

    pd.testing.assert_frame_equal(transformed_data, expected_data)
