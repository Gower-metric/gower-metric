# Gower-metric

Welcome to the Gower-metric library! This library provides an implementation of Gower's distance metric, which is particularly useful for calculating distances between samples with mixed data types 😁

## Table of Contents

- [Introduction](#introduction)
- [Compatibility matrix](#compatibility-matrix)
- [Documentation](#documentation)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Examples](#examples)
- [History](#history)
- [Citation](#citation)
- [References](#references)

## Introduction

Distance quantifies how far apart two objects are and is synonymous with dissimilarity. Calculating distance between individuals or groups is essential in fields like biology, psychology, ecology, medical diagnosis, and agriculture. It also underpins statistical methods like discriminant analysis, classification, and clustering, as well as machine learning algorithms such as k-nearest neighbor (supervised learning) and k-means clustering (unsupervised learning).

Euclidean distance is the standard measure for continuous variables, while the simple matching coefficient is common for categorical ones. However, real-world data often combines continuous and categorical variables (mixed data). Although extensive research exists for either continuous or categorical data, mixed data poses challenges. Researchers often either treat categorical data as continuous or transform continuous data into categorical, both of which can result in information loss.

To preserve data integrity, tailored formulas for mixed data types are necessary and one of them is Gower's similarity.

Implementation of Gower's Metric in Python.

## Compatibility matrix

| | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 | Python 3.14 |
|:---:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| Linux | ✅ | ✅ | ✅ | ✅ | ✅ |
| Windows | ✅ | ✅ | ✅ | ✅ | ✅ |
| macOS | ✅ | ✅ | ✅ | ✅ | ✅ |

Legend: ✅ - supported and tested, 🚧 - work in progress

## Documentation

Documentation is available [here](https://gower-metric.readthedocs.io/en/latest/).

## Installation

The easiest way to install the package is via pip:
```bash
pip install gower-metric
```

or via uv:
```bash
uv add gower-metric
```

## Quick start

Gower metric, by definition, is being designed to work with mixed, non-changed data types. We provide simple, two methods to how user can use algorithm. First, on original data. Second, on transformed ones (numerically encoded).

### Original data

In this scenario, user operates on imported, original and unchanged data.
```python
import numpy as np

from gower_metric import Config, Gower

data = np.array([[1, 'a', 3.5], [2, 'b', 4.0], [3, 'a', 2.5], [4, 'c', 5.0]], dtype=object)

feature_types = {
    0: "ratio_scale_interval",
    1: "categorical_nominal",
    2: "ratio_scale_interval"
}

cfg = Config(
    feature_types=feature_types,
)
gower = Gower(cfg).fit(data)
```

Finally run for two samples from the dataset:
```python
distance = gower(data[0], data[1])
```

### Transformation

We also provide a way to transform original data into it's numerical representation. It can be very useful when working with external libraries, 
such as scikit-learn and scipy. More about how to use them with gower-metric can be found within [examples](examples) folder and documentation.

```python
import numpy as np

from gower_metric import Config, Gower

data = np.array([[1, 'a', 3.5], [2, 'b', 4.0], [3, 'a', 2.5], [4, 'c', 5.0]], dtype=object)

feature_types = {
    0: "ratio_scale_interval",
    1: "categorical_nominal",
    2: "ratio_scale_interval"
}

cfg = Config(
    feature_types=feature_types,
)

gower = Gower(cfg).fit(data)
```

Now here is a difference. You can call `transform` method on class object as follows:

```python
transformed_data = gower.transform(data)
```

Under the hood, we map and convert all non-numerical values into numerical representation. With that in mind, all components calculated during fitting stage are being adjusted to transformed data and later used during distance calculation.

> [!IMPORTANT]
> Note that after calling `transform` method, user should NOT calculate components on original data anymore! This may lead to incorrect results.

Calculating distance between two samples is the same:

```python
distance = gower(transformed_data[0], transformed_data[1])
```

Why should you use transformed data? The main reason is compatibility with external libraries, that does not support gower metric natively and
require only numerical input data.

### Examples

Besides above description and documentation, we provide easy to copy-paste jupyter notebooks under the [examples](examples) folder.

## Contribution

Please refer to the [CONTRIBUTING](.github/CONTRIBUTING.md) file.

## History

@TODO: add history of this project

## Citation

If you find this project useful in your research, please consider citing it as follows:

@TODO: add citation

## References
- [Distances with mixed type variables some modified Gower's coefficients (2021)](https://arxiv.org/abs/2101.02481)
- [Extending Gower's General Coefficient of Similarity to Ordinal Characters (1999)](https://www.researchgate.net/publication/271789313_Extending_Gower%27s_General_Coefficient_of_Similarity_to_Ordinal_Characters)
- [A General Coefficient of Similarity and Some of Its Properties (1971)](https://www.jstor.org/stable/2528823)