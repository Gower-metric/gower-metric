# Description
In this section, we will provide quick comparisons of our Python approach compared to R environment. For time writing this section (2025-06),
only two R libraries are available:
- [gower](https://github.com/markvanderloo/gower)
- [daisy module](https://www.rdocumentation.org/packages/cluster/versions/2.1.8.1/topics/daisy)

Used data can be found in `comparison/data` directory. Please keep in mind that following packages are mostly based on [original Gower's algorithm from 1971](https://www.jstor.org/stable/2528823) without additional features mentioned in [paper from 2021](https://arxiv.org/abs/2101.02481). Used scripts can be found in `comparison/scripts/` directory.

## CRAN gower
Part of [CRAN project](https://cran.r-project.org/web/packages/gower/index.html), based on C language using openMP for parallelization. Detailed step-by-step guide can be found [here](https://cran.r-project.org/web/packages/gower/vignettes/intro.pdf). It is worth to mention that CRAN gower returns distance which corresponds to `1 - similarity`.

We made comparison on two subsets created from `iris.csv` file. First contains rows 0-9, second 5-14. Then we compare then accordingly, row 0 with row 5, row 1 with row 6, etc. The result is list with 10 elements. 

Environment | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRAN gower | 0.34606061 | 0.17939394 | 0.14303030 | 0.09636364 | 0.20424242 | 0.23636364 | 0.16000000 | 0.19939394 | 0.19818182 | 0.45030303 |
| Python | 0.34606061 | 0.17939394 | 0.14303030 | 0.09636364 | 0.20424242 | 0.23636364 | 0.16000000 | 0.19939394 | 0.19818182 | 0.45030303 |

Every row contains 4 numeric and 1 categorical variable. Overall range is calculated on both sets. Scale method is set to `range`. In original algorithm, missing data are not mentioned, thus authors skip them.

## Daisy gower

