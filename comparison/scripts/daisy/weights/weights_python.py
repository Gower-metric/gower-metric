import pandas as pd
import numpy as np

from gower_metric import Gower

df = pd.read_csv("comparison/data/weights_custom.csv")

# income as target, 1 if income > 40000, 0 otherwise -> binary symmetric
feature_types = {
    "age": "ratio_scale_interval",
    "gender": "categorical_nominal",
    "education": "categorical_ordinal",
    "hours_per_week": "numeric",
    "income": "binary_symmetric",
    "infected": "binary_asymmetric",
}

feature_weights = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
}

gower = Gower(feature_types, feature_weights=feature_weights).fit(df)

n = len(df)
matrix = np.zeros((n, n), dtype=np.float32)

for i in range(n):
    for j in range(n):
        matrix[i, j] = gower(df.iloc[i], df.iloc[j])

np.savetxt("comparison/scripts/daisy/weights/results/weights_python.txt", matrix, fmt="%.6f")