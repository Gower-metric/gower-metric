import numpy as np
import pytest

from gower_metric import Gower


def test_categorical_ordinal_kaufman_uniform_ndarray() -> None:
    data = np.array([["low"], ["medium"], ["high"], ["low"]], dtype=object)
    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
        0: ["low", "medium", "high"],
    }

    gower = Gower(
        {0: "categorical_ordinal"},
        categorical_ordinal_values_order=categorical_ordinal_values_order,
    )
    gower.fit(data)

    expected = np.array(
        [
            [0.0, 0.5, 1.0, 0.0],
            [0.5, 0.0, 0.5, 0.5],
            [1.0, 0.5, 0.0, 1.0],
            [0.0, 0.5, 1.0, 0.0],
        ],
        dtype=float,
    )

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            xi = data[i]
            xj = data[j]
            dist = gower(xi, xj)
            assert pytest.approx(dist, rel=1e-6) == expected[i, j]


def test_categorical_ordinal_podani_uniform_ndarray() -> None:
    data = np.array([["low"], ["medium"], ["high"], ["low"]], dtype=object)
    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
        0: ["low", "medium", "high"],
    }
    gower = Gower(
        {0: "categorical_ordinal"},
        categorical_ordinal_values_order=categorical_ordinal_values_order,
        categorical_ordinal_calculation_type="podani",
    )
    gower.fit(data)

    expected = np.array(
        [
            [0.0, 1 / 3, 1.0, 0.0],
            [1 / 3, 0.0, 2 / 3, 1 / 3],
            [1.0, 2 / 3, 0.0, 1.0],
            [0.0, 1 / 3, 1.0, 0.0],
        ],
        dtype=float,
    )

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            xi = data[i]
            xj = data[j]
            dist = gower(xi, xj)
            assert pytest.approx(dist, rel=1e-6) == expected[i, j]


def test_categorical_ordinal_not_valid_uniform_ndarray_() -> None:
    data = np.array([["low"], ["medium"], ["high"], ["low"]], dtype=object)

    with pytest.raises(ValueError):
        gower = Gower(
            {0: "categorical_ordinal"}, categorical_ordinal_calculation_type="not_valid"
        )
        gower.fit(data)


def test_categorical_ordinal_no_values_order_def_for_all_columns_() -> None:
    data = np.array(
        [["low", "high"], ["medium", "high"], ["high", "high"], ["low", "high"]],
        dtype=object,
    )
    categorical_ordinal_values_order: dict[int | str, list[str]] | None = {
        0: ["low", "medium", "high"],
    }

    with pytest.raises(ValueError):
        gower = Gower(
            {0: "categorical_ordinal", 1: "categorical_ordinal"},
            categorical_ordinal_values_order=categorical_ordinal_values_order,
        )
        gower.fit(data)
