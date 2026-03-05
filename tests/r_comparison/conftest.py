"""Conftest for R comparison tests."""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import (
    EDUCATION_LEVELS,
    GENDERS,
    RACES,
    generate_mixed_df,
    generate_numeric_with_categorical_df,
)

__all__ = [
    "EDUCATION_LEVELS",
    "generate_adult_like_df",
    "generate_mixed_df",
    "generate_numeric_with_categorical_df",
]


def generate_adult_like_df(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate adult-like DataFrame with 5 columns (R comparison format).

    This version keeps the original column names (education_num, sex, hours_per_week)
    for compatibility with R daisy() comparison tests.

    Args:
        n (int): Number of rows.
        rng (np.random.Generator): NumPy random generator instance.

    Returns:
        pd.DataFrame: DataFrame with 3 ratio and 2 categorical nominal columns.

    """
    df = pd.DataFrame(
        {
            "age": np.clip(rng.normal(38, 13, n), 17, 90).astype(int),
            "education_num": rng.integers(1, 17, n),
            "race": rng.choice(RACES, n, p=[0.7, 0.12, 0.08, 0.05, 0.05]),
            "sex": rng.choice(GENDERS, n),
            "hours_per_week": np.clip(rng.normal(40, 12, n), 1, 99).astype(int),
        },
    )
    df["race"] = df["race"].astype("category")
    df["sex"] = df["sex"].astype("category")
    return df


@pytest.fixture
def random_seed() -> int:
    """Provide a static seed."""
    return 42
