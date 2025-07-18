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

> [!Important]
> Please do remember, there is no automatic feature type detection. One must be aware of data they use and provide. Any potential mistakes could occur due to mislabeling feature types, so please be careful and double-check your input data types.

## Input data type - format
There are two supported input data types.

### Numpy array
```python
import numpy as np

from gower_similarity.core.similarity import GowerSimilarity

data = np.array([[1], [4], [7]], dtype=float)
f_types = {
    0: "ratio_scale_interval",
}
gs = GowerSimilarity(f_types).fit(data)
```
> [!Note]
> Using numpy array with no numerical data, please set `dtype = object` to avoid issues with data types.

### Pandas DataFrame
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
> [!Tip]
> To make matrix based on similarity, just use `gs.similarity` instead of `gs.distance`.

## Advanced usage
We also provide basic weighting functionality. You can set weights for each feature type in the `GowerSimilarity` constructor. The weights should be provided as a dictionary where keys are columns indexes and values are weights. Example script can be found in `examples/scripts/weight.py`.

### Weights
To set weights, you can create a dictionary and pass it to the `GowerSimilarity` constructor as follows:

```python
import pandas as pd

from gower_similarity.core.similarity import GowerSimilarity

df = pd.read_csv("comparison/data/adult_reduced.csv").head(5)

feature_types = {
    "age": "ratio_scale_interval",
    "education_num": "ratio_scale_interval",
    "race": "categorical_nominal",
    "sex": "categorical_nominal",
    "hours_per_week": "ratio_scale_interval",
}

feature_weights = {
    0: 1.0,
    1: 2.0,
    2: 3.0,
    3: 4.0,
    4: 5.0,
}

gs = GowerSimilarity(feature_types, feature_weights=feature_weights).fit(df)
```

Please note that specific values are assigned to column index, not name.

### Compatibility with other libraries

For time writing this section, there is no full compatibility with other libraries. However, one can use our custom API to calculate similarity and distance between rows, as well as create a matrix. In `examples/scripts/scktlrn_knn.ipynb` we provide quick guide how to integrate our library with scikit-learn kNN algorithm. What is more, we also provide the script with explanation how to use our library with scikit-learn HDBSCAN algorithm in `examples/scripts/scktlrn_hdbscan.ipynb`.

### Scikit-learn pairwise_distances

One would want to simplify the process of creating a matrix using scikit-learn's `pairwise_distances` function. It is possible to achieve that, however it requires some additional steps.

```python
from sklearn.metrics import pairwise_distances

# after we fit our gower, we have to create additional function to compute distance
def _gower_distance(x, y):
    """
    Compute Gower distance between two vectors.
    """
    return gs.distance(x, y)

matrix = pairwise_distances(df, metric = _gower_distance, n_jobs = -1, ensure_all_finite = False)
```
> [!Warning]
> It is worth to mention that one of gower's idea is to handle missing data as user wants thus `ensure_all_finite = False` is required to allow missing data in the input DataFrame. 

Full code can be found in `examples/scripts/scktlrn_pairwise_distances.ipynb`.

### SciPy spatial distance

The idea is very similar to the one used in scikit-learn. You can use `scipy.spatial.distance.pdist` to calculate distance between rows.
```python
from scipy.spatial.distance import pdist, squareform

def _gower_distance(x, y):
        """
        Compute Gower distance between two vectors.
        """
        return gs.distance(x, y)
    
array_scipy = pdist(df, metric = _gower_distance)
matrix_scipy = squareform(array_scipy)
```

Script can be found in `examples/scripts/scipy_pdist.ipynb`.

### Why use joblib?

Let's go back to `adult_reduced.csv` example and reduce dataset to first 5 000 rows. If you want to calculate similarity for all rows, it can take a while. To speed up the process, you can use `joblib` to parallelize the computation.

No joblib | joblib | joblib half matrix |
| :---: | :---: | :---: |
| 1199.16 s | 169.11 s | 86.39 s |

Used script can be found in `examples/scripts/matrix_speedup.py`.