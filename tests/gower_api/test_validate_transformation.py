import pandas as pd
import pytest

from gower_metric import Gower
from gower_metric.core.config import Config
from gower_metric.core.exceptions import IllegalStateError


def test_validate_transformation_pandas() -> None:
    data = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "gender": ["male", "female", "male", "female", "male"],
            "income": [50000, 60000, 70000, 80000, 90000],
        },
    )
    ft: dict[str | int, str] = {
        "age": "ratio_scale_interval",
        "gender": "categorical_nominal",
        "income": "ratio_scale_interval",
    }
    cfg = Config(feature_types=ft)
    gower = Gower(cfg)
    gower.fit(data)
    _ = gower.transform(data)

    with pytest.raises(IllegalStateError):
        gower(data.iloc[0], data.iloc[1])


def test_validate_double_transformed() -> None:
    data = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "gender": ["male", "female", "male", "female", "male"],
            "income": [50000, 60000, 70000, 80000, 90000],
        },
    )
    ft: dict[str | int, str] = {
        "age": "ratio_scale_interval",
        "gender": "categorical_nominal",
        "income": "ratio_scale_interval",
    }
    cfg = Config(feature_types=ft)
    gower = Gower(cfg)
    gower.fit(data)
    _ = gower.transform(data)

    with pytest.raises(IllegalStateError):
        gower.transform(data)
