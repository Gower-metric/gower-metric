# TODO list for refactor

## Different data types:
- ~~categorical nominal~~
- ~~categorical ordinal~~
- ~~binary symmetric~~
- ~~binary asymmetric~~
- ~~ratio scale~~
- ~~numerical interval~~

## Handling NaN values:
- raise
- ~~ignore~~
- max_dist
  
For now, we ignore NaN values by default.

## Weights:
- ~~manually set (dictionary)~~
- precomputed (+ load from file)
- save to file
- CPCC 

## Range scale method:
- ~~range~~
- ~~iqr~~
- kde

## Pytest:
- ~~categorical nominal~~
- categorical ordinal
- ~~binary symmetric~~
- ~~binary asymmetric~~
- ~~ratio scale~~
- numerical interval
- ~~missing values~~
- mixed types

## Examples:
- categorical nominal
- categorical ordinal
- binary symmetric
- binary asymmetric
- ratio scale
- numerical interval
- mixed types

## Additional features:
- normalization
- Podani for categorical ordinal
- Podani weight optimization
- add automatic check if given data type is correct
- ~~in numeric/ratio scale add ranges as function parameter rather than compute it every time~~
- fix weights in count_present