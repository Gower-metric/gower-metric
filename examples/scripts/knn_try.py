import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from gower_metric import Gower
from gower_metric.core.config import Config

warnings.filterwarnings("ignore")

import os
print("CWD:", os.getcwd())


# load and preprocess data
n_rows = 1000
# df = pd.read_csv("comparison/data/adult.csv",
#     dtype={
#         "age": "int64",
#         "educational-num": "int64",
#         "race": "string",
#         "gender": "string",
#         "hours-per-week": "int64",
#         "relationship": "string",
#         "occupation": "string",
#         "education": "string",
#         "workclass": "string"
#     }
# ).head(n_rows)

df = pd.read_csv("data/files/adult.csv", dtype=None).head(n_rows)

feature_cols = [
    "age",
    "educational-num",
    "race",
    "gender",
    "hours-per-week",
    "relationship",
    "occupation",
    "education",
    "workclass",
]
target_col = "income"

# convert income to 0 if <=50K, 1 if >50K
X = df[feature_cols]
y = df[target_col].apply(lambda x: 1 if x == ">50K" else 0)

X.replace(to_replace="?", value=np.nan)

# define features types
feature_types : dict[int | str, str] = {
    "age": "ratio_scale_interval",
    "educational-num": "ratio_scale_interval",
    "race": "categorical_nominal",
    "gender": "categorical_nominal",
    "hours-per-week": "ratio_scale_interval",
    "relationship": "categorical_nominal",
    "occupation": "categorical_nominal",
    "education": "categorical_nominal",
    "workclass": "categorical_nominal",
}
cfg = Config(
    feature_types=feature_types,
    conditional_distances=False,
)
gower = Gower(cfg).fit(X)
X_transformed = gower.transform(X)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42, stratify=y
)


knn = KNeighborsClassifier(n_neighbors=5, metric=gower)

knn.fit(X_train, y_train)