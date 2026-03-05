import os
import sys

import numpy as np
import pandas as pd

# assuming all main files and imports are one level above "tests" subfolder
main_repo_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# if not present, add it
if main_repo_root not in sys.path:
    sys.path.insert(0, main_repo_root)

TYPES_TO_CHECK = [
    "float16",
    "float32",
    "float64",
    "float128",
]  # here define all numerical types to check
NUMPY_NUMERIC_TYPES = [getattr(np, t) for t in TYPES_TO_CHECK if hasattr(np, t)]

EDUCATION_LEVELS = ["Low", "Medium", "High", "PhD", "Prof"]
COUNTRIES = ["PL", "DE", "FR", "UK", "US", "IT", "ES", "CZ"]
GENDERS = ["Female", "Male"]
RACES = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
RELATIONSHIPS = [
    "Husband",
    "Not-in-family",
    "Wife",
    "Own-child",
    "Unmarried",
    "Other-relative",
]
OCCUPATIONS = [
    "Prof-specialty",
    "Craft-repair",
    "Exec-managerial",
    "Adm-clerical",
    "Sales",
    "Other-service",
    "Machine-op-inspct",
    "Transport-moving",
    "Handlers-cleaners",
    "Farming-fishing",
]
EDUCATION_CATEGORIES = [
    "HS-grad",
    "Some-college",
    "Bachelors",
    "Masters",
    "Assoc-voc",
    "11th",
    "Assoc-acdm",
    "10th",
    "Doctorate",
    "7th-8th",
]
WORKCLASSES = [
    "Private",
    "Self-emp-not-inc",
    "Local-gov",
    "State-gov",
    "Self-emp-inc",
    "Federal-gov",
]


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
        pd.DataFrame: DataFrame with 3 ratio and 6 categorical nominal columns.

    """
    df = pd.DataFrame(
        {
            "age": np.clip(rng.normal(38, 13, n), 17, 90).astype(int),
            "educational-num": rng.integers(1, 17, n),
            "hours-per-week": np.clip(rng.normal(40, 12, n), 1, 99).astype(int),
            "race": rng.choice(RACES, n, p=[0.7, 0.12, 0.08, 0.05, 0.05]),
            "gender": rng.choice(GENDERS, n),
            "relationship": rng.choice(RELATIONSHIPS, n),
            "occupation": rng.choice(OCCUPATIONS, n),
            "education": rng.choice(EDUCATION_CATEGORIES, n),
            "workclass": rng.choice(WORKCLASSES, n),
        },
    )
    for col in [
        "race",
        "gender",
        "relationship",
        "occupation",
        "education",
        "workclass",
    ]:
        df[col] = df[col].astype("category")
    return df
