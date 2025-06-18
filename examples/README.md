# Description
In this section, we provide examples of how to use our API to perform data analysis. If you have any other questions or need further assistance, please feel free to reach out to us via GitHub Issues section.

## Understanding feature types
To make library work properly, user HAS TO provide accurate feature types. Long story short, every column from input data can be labeled as one of the following types:
- categorical_nominal
- categorical_ordinal
- binary_asymmetric
- binary_symmetric
- ratio_scale_interval
- numeric

Please do remember, there is no automatic feature type detection. One must be aware of data they use and provide. Any potential mistakes could occur due to mislabeling feature types, so please be careful and double-check your input data types.

## Input data type - format
There are two supported input data types:
- numpy array: np.array
```python
import numpy as np

from gower_similarity.core.similarity import GowerSimilarity

data = np.array([[1], [4], [7]], dtype=float)
f_types = {
    0: "ratio_scale_interval",
}
gs = GowerSimilarity(f_types).fit(data)
```
Note: using numpy array with no numerical data, please set `dtype=object` to avoid issues with data types.
- pandas DataFrame: pd.DataFrame
```python
import pandas as pd

from gower_similarity.core.similarity import GowerSimilarity

iris = pd.read_csv("path/iris.csv")

f_types = {
    "sepal_length": "numeric",
    "sepal_width": "numeric",
    "petal_length": "numeric",
    "petal_width": "numeric",
    "variety": "categorical_nominal",
}

gs = GowerSimilarity(f_types).fit(iris)
```
Do not worry, under the hood, we convert pandas DataFrame to numpy array, so you can use the same API for both data types.

## Calculating similarity and distance between rows
Created API allows you to calculate similarity and distance between rows using separate methods.

```python
import numpy as np

from gower_similarity.core.similarity import GowerSimilarity

data = np.array([['A'], ['B'], ['C'], ['A']], dtype=object)
gs = GowerSimilarity({0: 'categorical_nominal'}).fit(data)

row_0 = data[0]
row_3 = data[3]

similarity = gs.similarity(row_0, row_3)
distance = gs.distance(row_0, row_3)
```

## Creating matrix
For now, there is no API endpoint to create whole matrix using provided data. However, you can do it manually using previous methods in a loop.

```python
import pandas as pd
import numpy as np

from gower_similarity.core.similarity import GowerSimilarity

df = pd.DataFrame({
    "age": [23, 45, 23, 31],
    "gender": ["Female", "Male", "Female", "Male"],
    "income": [35000, 81000, 40000, 30000],
    "education": ["low", "medium", "high", "low"],
    "married": [0, 1, 1, 0],
    "infected": [1, 1, 0, 0],
})

feature_types = {
    "age": "ratio_scale_interval",
    "gender": "categorical_nominal",
    "income": "numeric",
    "education": "categorical_ordinal",
    "married": "binary_symmetric",
    "infected": "binary_asymmetric",
}

gs = GowerSimilarity(feature_types, scale="range").fit(df)

n = len(df)
matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        matrix[i, j] = gs.distance(df.iloc[i], df.iloc[j])
```
To make matrix based on similarity, just use `gs.similarity` instead of `gs.distance`.