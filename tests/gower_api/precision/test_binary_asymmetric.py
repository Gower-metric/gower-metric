import numpy as np
import pandas as pd
import pytest

from gower_metric import Config, Gower
from tests.gower_api.precision.conftest import BaseTest


class TestBinaryAsymmetric(BaseTest):
    def test_binary_asymmetric_only(self) -> None:
        data = np.array([[0], [1], [0]], dtype=object)
        cfg = Config(
            feature_types={0: "binary_asymmetric"},
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)

        # 0/0: NaN
        assert np.isnan(gower(data[0], data[2]))

        # 1/1: distance 0
        assert pytest.approx(gower(data[1], data[1]), rel=1e-6) == 0.0

        # 1/0: distance 1
        assert pytest.approx(gower(data[1], data[0]), rel=1e-6) == 1.0

    def test_binary_asymmetric_only_pandas(self) -> None:
        data = pd.DataFrame({"column": [0, 1, 0]})
        cfg = Config(
            feature_types={"column": "binary_asymmetric"},
            data_type=self.dtype,
        )
        gower = Gower(cfg).fit(data)

        assert np.isnan(gower(data.iloc[0], data.iloc[2]))
        assert pytest.approx(gower(data.iloc[1], data.iloc[1]), rel=1e-6) == 0.0
        assert pytest.approx(gower(data.iloc[1], data.iloc[0]), rel=1e-6) == 1.0
