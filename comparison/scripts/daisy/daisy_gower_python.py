import numpy as np
import pandas as pd

from gower_metric import Gower

iris = pd.read_csv("your_path/iris.csv")

f_types: dict[int | str, str] = {
    "sepal_length": "numeric",
    "sepal_width": "numeric",
    "petal_length": "numeric",
    "petal_width": "numeric",
    "variety": "categorical_nominal",
}

gower = Gower(f_types, scale="range").fit(iris)

rows_1 = range(10)
rows_2 = range(5, 15)

py_block = np.zeros((10, 10))
for i, r1 in enumerate(rows_1):
    for j, r2 in enumerate(rows_2):
        py_block[i, j] = gower(iris.iloc[r1], iris.iloc[r2])

# prepare columns for R-like indexing, pure demonstration purpose
col6 = py_block[:, 0]  # 6-th column corresponds to row 6 in R indexing
col7 = py_block[:, 1]  # 7-th column corresponds to row 7 in R indexing

print("col6:", col6)
print("col7:", col7)
