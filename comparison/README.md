# Description
In this section, we will provide quick comparisons of our Python approach compared to R environment. For time writing this section (2025-06),
only two R libraries are available:
- [gower](https://github.com/markvanderloo/gower)
- [daisy module](https://www.rdocumentation.org/packages/cluster/versions/2.1.8.1/topics/daisy)

Used data can be found in [data/files](../data/files/) directory. Please keep in mind that following packages are mostly based on [original Gower's algorithm from 1971](https://www.jstor.org/stable/2528823) without additional features mentioned in paper [from 1999](https://www.researchgate.net/publication/271789313_Extending_Gower%27s_General_Coefficient_of_Similarity_to_Ordinal_Characters) or [from 2021](https://arxiv.org/abs/2101.02481). Used scripts can be found in [scripts](./scripts/) directory.

## CRAN gower
Part of [CRAN project](https://cran.r-project.org/web/packages/gower/index.html), based on C language using openMP for parallelization. Detailed step-by-step guide can be found [here](https://cran.r-project.org/web/packages/gower/vignettes/intro.pdf). It is worth to mention that CRAN gower returns distance which corresponds to `1 - similarity`.

We made comparison on two subsets created from [iris.csv](../data/files/iris.csv) file. First contains rows 0-9, second 5-14. Then we compare then accordingly, row 0 with row 5, row 1 with row 6, etc. The result is list with 10 elements. 

<details>
<summary><h3 style="display:inline; margin:0;">Results</h3></summary>

| row | CRAN gower |   Python   |
|:---:|:----------:|:----------:|
| 0   | 0.34606061 | 0.34606061 |
| 1   | 0.17939394 | 0.17939394 |
| 2   | 0.14303030 | 0.14303030 |
| 3   | 0.09636364 | 0.09636364 |
| 4   | 0.20424242 | 0.20424242 |
| 5   | 0.23636364 | 0.23636364 |
| 6   | 0.16000000 | 0.16000000 |
| 7   | 0.19939394 | 0.19939394 |
| 8   | 0.19818182 | 0.19818182 |
| 9   | 0.45030303 | 0.45030303 |

</details>

Every row contains 4 numeric and 1 categorical data type. Overall range is calculated on both sets, combined. Scale method is set to `range`. In original algorithm, missing data are not mentioned, thus authors skip them. They are not present in our data.


## Daisy gower

<details>
<summary><h3 style="display:inline; margin:0;">Results</h3></summary>

| row | Daisy Gower (6) | Daisy Gower (7) | Python (6)      | Python (7)      |
|:---:|:---------------:|:---------------:|:---------------:|:---------------:|
| 1   | 0.07683616      | 0.04444444      | 0.07683616      | 0.04444444      |
| 2   | 0.12961394      | 0.05833333      | 0.12961394      | 0.05833333      |
| 3   | 0.12744821      | 0.03394539      | 0.12744821      | 0.03394539      |
| 4   | 0.13455744      | 0.03672316      | 0.13455744      | 0.03672316      |
| 5   | 0.07405838      | 0.04722222      | 0.07405838      | 0.04722222      |
| 6   | 0.00000000      | 0.10461394      | 0.00000000      | 0.10461394      |
| 7   | 0.10461394      | 0.00000000      | 0.10461394      | 0.00000000      |
| 8   | 0.08733522      | 0.03394539      | 0.08733522      | 0.03394539      |
| 9   | 0.16572505      | 0.06111111      | 0.16572505      | 0.06111111      |
| 10  | 0.12622411      | 0.06172316      | 0.12622411      | 0.06172316      |

</details>

In example above, we compare the same dataset using daisy module from R and our Python implementation. Please be aware, that rows ids in R are 1-based compared to 0-based Python. Results are different due to calculating range scaling on all data not only first 20 rows. 

## Speed
Here we compare speed between our Python and R implementation. All calculation are made on [adult_reduced.csv](../data/files/adult_reduced.csv) file, for 100 iterations.

### CRAN gower
Firstly, we consider cran gower based on openMP C implementation. We compare first row with second, second with third, etc. Results below.

<details>
<summary><h4 style="display:inline; margin:0;">Img</h4></summary>

![CRAN gower speed comparison](../data/imgs/comparison/cran_gower_similarity_comparison.png)

</details>

### Daisy gower
In contrary, using daisy module we can compare all rows with each other, which creates a matrix of distance. In our case, we reduced used data to 1000 rows, due to pure Python optimization (for now). Results below.

<details>
<summary><h4 style="display:inline; margin:0;">Img</h4></summary>

![Daisy gower speed comparison](../data/imgs/comparison/daisy_gower_distance_comparison.png)

</details>

## Weights and handling NaN values

In addition to the above comparisons, we also tested how our Python implementation handles weights and NaN values compared to the daisy module in R. Both scripts can be found in the [scripts/daisy/weights](./scripts/daisy/weights/) and [scripts/daisy/nan_values](./scripts/daisy/nan_values/) directories. Results below.

### Weights

Firstly, we tested the effect of applying weights to different features in the dataset. We assigned higher weights to certain features and observed how this influenced the results. Calculated distances matrices from both implementations can be seen below.

<details>
<summary><h4 style="display:inline; margin:0;">Results</h4></summary>

- Python
  
|   |     1    |     2    |     3    |    4     |    5     | 
|:-:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 1 | 0.000000 | 0.661968 | 0.545306 | 0.718685 | 0.754667 |
| 2 | 0.661968 | 0.000000 | 0.798186 | 0.542993 | 0.199365 |
| 3 | 0.545306 | 0.798186 | 0.000000 | 0.588526 | 0.839456 |
| 4 | 0.718685 | 0.542993 | 0.588526 | 0.000000 | 0.452744 |
| 5 | 0.754667 | 0.199365 | 0.839456 | 0.452744 | 0.000000 |

- R Daisy
  
|   |     1    |     2    |     3    |    4     |    5     | 
|:-:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 1 | 0.000000 | 0.661968 | 0.545306 | 0.718685 | 0.754667 |
| 2 | 0.661968 | 0.000000 | 0.798186 | 0.542993 | 0.199365 |
| 3 | 0.545306 | 0.798186 | 0.000000 | 0.588526 | 0.839456 |
| 4 | 0.718685 | 0.542993 | 0.588526 | 0.000000 | 0.452744 |
| 5 | 0.754667 | 0.199365 | 0.839456 | 0.452744 | 0.000000 |

</details>

### Handling NaN values

Next, we evaluated how both implementations handle NaN values in the dataset. We introduced NaN values in various positions, set methodology to `ignore` and compared the resulting distance matrices. Results from both implementations are shown below.

<details>
<summary><h4 style="display:inline; margin:0;">Results</h4></summary>

- Python
  
|   |     1    |     2    |     3    |     4    |     5    |     6    |     7    |
|:-:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 1 | 0.000000 | 0.649524 | 0.571679 | 0.522449 | 0.713605 | 0.751111 | 0.446667 |
| 2 | 0.649524 | 0.000000 | 0.695489 | 0.766440 | 0.529025 | 0.190476 | 0.625000 |
| 3 | 0.571679 | 0.695489 | 0.000000 | 0.384712 | 0.385464 | 0.687970 | 0.203704 |
| 4 | 0.522449 | 0.766440 | 0.384712 | 0.000000 | 0.570748 | 0.814059 | 0.166667 |
| 5 | 0.713605 | 0.529025 | 0.385464 | 0.570748 | 0.000000 | 0.445125 | 0.573333 |
| 6 | 0.751111 | 0.190476 | 0.687970 | 0.814059 | 0.445125 | 0.000000 | 0.733333 |
| 7 | 0.446667 | 0.625000 | 0.203704 | 0.166667 | 0.573333 | 0.733333 | 0.000000 |

- R Daisy
  
|   |     1    |     2    |     3    |     4    |     5    |     6    |     7    |
|:-:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 1 | 0.000000 | 0.649524 | 0.571679 | 0.522449 | 0.713605 | 0.751111 | 0.446667 |
| 2 | 0.649524 | 0.000000 | 0.695489 | 0.766440 | 0.529025 | 0.190476 | 0.625000 |
| 3 | 0.571679 | 0.695489 | 0.000000 | 0.384712 | 0.385464 | 0.687970 | 0.203704 |
| 4 | 0.522449 | 0.766440 | 0.384712 | 0.000000 | 0.570748 | 0.814059 | 0.166667 |
| 5 | 0.713605 | 0.529025 | 0.385464 | 0.570748 | 0.000000 | 0.445125 | 0.573333 |
| 6 | 0.751111 | 0.190476 | 0.687970 | 0.814059 | 0.445125 | 0.000000 | 0.733333 |
| 7 | 0.446667 | 0.625000 | 0.203704 | 0.166667 | 0.573333 | 0.733333 | 0.000000 |

</details>

## Conclusion

Long story short, it is safe to assume that our implementation is numerically consistent with R's implementation.

Files with results can be found in [daisy/weights/weights_results](./scripts/daisy/weights/weights_results/) and [daisy/nan_values/nan_values_results](./scripts/daisy/nan_values/nan_values_results/) directories respectively.