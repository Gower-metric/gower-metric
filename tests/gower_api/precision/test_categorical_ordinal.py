import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gower_metric import Config, Gower
from tests.gower_api.precision.conftest import BaseTest


class TestCategoricalOrdinal(BaseTest):
    def test_categorical_ordinal_kaufman_uniform_ndarray(self) -> None:
        data = np.array([["low"], ["medium"], ["high"], ["low"]], dtype=object)
        categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
            0: ["low", "medium", "high"],
        }

        cfg = Config(
            feature_types={0: "categorical_ordinal"},
            data_type=self.dtype,
            categorical_ordinal_values_order=categorical_ordinal_values_order,
        )
        gower = Gower(cfg).fit(data)

        expected = np.array(
            [
                [0.0, 0.5, 1.0, 0.0],
                [0.5, 0.0, 0.5, 0.5],
                [1.0, 0.5, 0.0, 1.0],
                [0.0, 0.5, 1.0, 0.0],
            ],
            dtype=self.dtype,
        )

        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                dist = gower(data[i], data[j])
                assert pytest.approx(dist, rel=1e-6) == expected[i, j]

    def test_categorical_ordinal_podani_uniform_ndarray(self) -> None:
        data = np.array([["low"], ["medium"], ["high"], ["low"]], dtype=object)
        categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
            0: ["low", "medium", "high"],
        }

        cfg = Config(
            feature_types={0: "categorical_ordinal"},
            data_type=self.dtype,
            categorical_ordinal_values_order=categorical_ordinal_values_order,
            categorical_ordinal_calculation_type="podani",
        )
        gower = Gower(cfg).fit(data)

        PODANI_EXPECTED_RESULTS = np.array(
            [
                [0.0, 1 / 3, 1.0, 0.0],
                [1 / 3, 0.0, 2 / 3, 1 / 3],
                [1.0, 2 / 3, 0.0, 1.0],
                [0.0, 1 / 3, 1.0, 0.0],
            ],
            dtype=self.dtype,
        )

        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                dist = gower(data[i], data[j])
                assert pytest.approx(dist, rel=1e-6) == PODANI_EXPECTED_RESULTS[i, j]

    def test_categorical_ordinal_podani_uniform_df(self) -> None:
        data = pd.DataFrame(
            np.array([["low"], ["medium"], ["high"], ["low"]], dtype=object),
            columns=["level"],
        )
        categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
            "level": ["low", "medium", "high"],
        }

        cfg = Config(
            feature_types={"level": "categorical_ordinal"},
            data_type=self.dtype,
            categorical_ordinal_values_order=categorical_ordinal_values_order,
            categorical_ordinal_calculation_type="podani",
        )
        gower = Gower(cfg).fit(data)

        transformed_data = gower.transform(data)
        if isinstance(transformed_data, pd.DataFrame):
            assert (transformed_data.dtypes == self.dtype).all()

        PODANI_EXPECTED_RESULTS = np.array(
            [
                [0.0, 1 / 3, 1.0, 0.0],
                [1 / 3, 0.0, 2 / 3, 1 / 3],
                [1.0, 2 / 3, 0.0, 1.0],
                [0.0, 1 / 3, 1.0, 0.0],
            ],
            dtype=self.dtype,
        )

        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                dist = gower(data.iloc[i], data.iloc[j])
                assert pytest.approx(dist, rel=1e-6) == PODANI_EXPECTED_RESULTS[i, j]

    def test_categorical_ordinal_not_valid_uniform_ndarray(self) -> None:
        with pytest.raises(ValidationError):
            Config(
                feature_types={"level": "categorical_ordinal"},
                categorical_ordinal_calculation_type="not_valid",  # type: ignore[arg-type]
            )

    def test_categorical_ordinal_no_values_order_def_for_all_columns(self) -> None:
        categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
            0: ["low", "medium", "high"],
        }

        with pytest.raises(ValidationError):
            Config(
                feature_types={0: "categorical_ordinal", 1: "categorical_ordinal"},
                categorical_ordinal_values_order=categorical_ordinal_values_order,
            )
