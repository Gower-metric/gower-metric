import numpy as np
import pytest

from gower_metric import Gower


def test_binary_asymmetric_only() -> None:
    data = np.array([[0], [1], [0]], dtype=object)
    gower = Gower({0: "binary_asymmetric"})
    gower.fit(data)

    # 0/0: NaN
    assert np.isnan(gower(data[0], data[2]))

    # 1/1: distance 0
    assert pytest.approx(gower(data[1], data[1]), rel=1e-6) == 0.0

    # 1/0: distance 1
    assert pytest.approx(gower(data[1], data[0]), rel=1e-6) == 1.0
