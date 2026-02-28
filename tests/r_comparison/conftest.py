"""Shared fixtures for R comparison tests."""

import logging

import numpy as np
import pandas as pd
import pytest

logger = logging.getLogger(__name__)


EDUCATION_LEVELS = ["Low", "Medium", "High", "PhD", "Prof"]
COUNTRIES = ["PL", "DE", "FR", "UK", "US", "IT", "ES", "CZ"]
GENDERS = ["Female", "Male"]
RACES = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]


def generate_mixed_df(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a DataFrame with all 6 feature types using statistical distributions.

    Args:
        n (int): Number of rows to generate.
        rng (np.random.Generator): NumPy random generator instance.

    Returns:
        pd.DataFrame: DataFrame with columns covering all Gower feature types:
            - Age (numeric, normal distribution)
            - Salary (ratio_scale_interval, exponential distribution)
            - Have_children (binary_symmetric, Bernoulli p=0.5)
            - Is_smoking (binary_asymmetric, Bernoulli p=0.3)
            - Birth (categorical_nominal, uniform choice)
            - Education (categorical_ordinal, weighted choice)

    """
    have_children = rng.choice([0, 1], n)
    is_smoking = rng.choice([0, 1], n, p=[0.7, 0.3])
    education = rng.choice(
        EDUCATION_LEVELS,
        n,
        p=[0.3, 0.3, 0.2, 0.15, 0.05],
    )

    have_children[0], have_children[1] = 0, 1
    is_smoking[0], is_smoking[1] = 0, 1
    for i, level in enumerate(EDUCATION_LEVELS):
        education[i] = level

    return pd.DataFrame(
        {
            "Age": np.clip(rng.normal(35, 12, n), 18, 80).astype(int),
            "Salary": np.round(rng.exponential(45000, n), 2),
            "Have_children": have_children,
            "Is_smoking": is_smoking,
            "Birth": rng.choice(COUNTRIES, n),
            "Education": education,
        },
    )


def generate_numeric_with_categorical_df(
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate a DataFrame similar to iris — mostly numeric + one categorical.

    Args:
        n (int): Number of rows.
        rng (np.random.Generator): NumPy random generator instance.

    Returns:
        pd.DataFrame: DataFrame with 4 numeric and 1 categorical column.

    """
    return pd.DataFrame(
        {
            "sepal_length": np.round(rng.normal(5.8, 0.8, n), 1),
            "sepal_width": np.round(rng.normal(3.0, 0.4, n), 1),
            "petal_length": np.round(rng.exponential(2.5, n), 1),
            "petal_width": np.round(rng.exponential(0.8, n), 1),
            "variety": rng.choice(["Setosa", "Versicolor", "Virginica"], n),
        },
    )


def generate_adult_like_df(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a DataFrame resembling adult.csv — ratio + categorical columns.

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
    """Provide a random seed and log it for reproducibility."""
    seed = int(np.random.default_rng().integers(0, 2**31))
    logger.info("Random seed for this test run: %d", seed)
    return seed
