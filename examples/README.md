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