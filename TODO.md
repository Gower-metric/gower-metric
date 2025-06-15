# TODO list for refactor

## Different data types:
- ~~categorical nominal~~
- ~~categorical ordinal~~
- ~~binary symmetric~~
- ~~binary asymmetric~~
- ~~ratio scale~~
- ~~numerical interval~~

## Handling NaN values:
- ~~raise~~
- ~~ignore~~
- ~~max_dist~~

## Weights:
- ~~manually set (dictionary), the same as "uniform"~~
- precomputed (+ load from file)
- save to file
- CPCC 

## Range scale method:
- ~~range~~
- ~~iqr~~
- kde: 
    - ~~silverman~~
    - scott
    - sheather-jones
    - cv_grid
    - cv_optuna
- ~~kNN~~

## Pytest:
- ~~categorical nominal~~
- ~~categorical ordinal~~
- ~~binary symmetric~~
- ~~binary asymmetric~~
- ~~ratio scale~~
  - ~~kde silverman~~
- numerical interval
  - kde silverman
- ~~missing values~~
- mixed types
- ~~kNN bandwidth~~

## Examples:
- categorical nominal
- categorical ordinal
- binary symmetric
- binary asymmetric
- ratio scale
- numerical interval
- mixed types

## Additional features:
- ~~normalization~~
- continous distances
- refactor calculate bandwidth for kde and kNN
- ~~Podani for categorical ordinal~~
- ~~Podani weight optimization~~
- ~~add automatic check if given data type is correct~~
- ~~in numeric/ratio scale add ranges as function parameter rather than compute it every time~~
- ~~fix weights in count_present~~
- numba speedup