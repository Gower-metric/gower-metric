# Gower-metric

Welcome to the Gower-metric library! This library provides an implementation of Gower's distance metric, which is particularly useful for calculating distances between samples with mixed data types 😁

## Table of Contents

- [Introduction](#introduction)
- [Compatibility](#compatibility)
- [Documentation](#documentation)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Gower characteristics](#gower-characteristics)
- [Metric enhancements](#metric-enhancements)
- [Metrics comparison](#metrics-comparison)
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

@TODO: add ref  
Documentation is available [here]().

## Examples

Beside documentation, we provide easy to copy-paste jupyter notebooks under the [examples](examples) folder.

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

In order to import class, which calculate Gower's metric, you need to import it as follows:
```python
from gower_metric import Gower
```

then initialize passing the variable types:
```python
data = [[1, 'a', 3.5], [2, 'b', 4.0], [3, 'a', 2.5], [4, 'c', 5.0]]

feature_types = {
    0: "ratio_scale_interval",
    1: "categorical_nominal",
    2: "ratio_scale_interval"
}

gower = Gower(feature_types=feature_types)
```

fit it to the data:
```python
gower.fit(data)
```

and finally run for two samples from the dataset:
```python
distance = gower(data[0], data[1])
```

> [!Tip]
> To calculate the pairwise distances for the entire dataset, you can do it manually or use an auxiliary function, like: [scipy.spatial.distance.pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) or [sklearn.metrics.pairwise_distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html).


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