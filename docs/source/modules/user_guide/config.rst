====================
Configuration Class
====================

Due to the Gower's algorithm idea, we have to provide additional information about used data. The ONLY required parameter is called
``feature_types``, which is a dictionary that maps each feature index (column index or column name) to its corresponding type.

-----------
Data types
-----------

Why should I care about it? One could assume that there should be some kind of automatic detection of data types. Well, yes and no.
Automatic detection is being applied to scipy and scikit-learn libraries, but it is limited to a subset of data types. Numerical and
categorical nominal data can be detected automatically, but that's about it. Thus, we provide a more extensive capability to specify data types
manually. Below is the list of supported categories:

- ``categorical_nominal`` – categorical data without any order or ranking among values. E.g., colors, names, etc.
- ``categorical_ordinal`` – categorical data with order or ranking among values. E.g., ratings (bad, average, good), education levels (low, medium, high), etc.
- ``binary_symmetric`` – binary data where both outcomes are equally important. E.g., gender (male/female), yes/no questions, etc.
- ``binary_asymmetric`` – binary data where one outcome is more significant than the other. E.g., presence/absence of a disease, success/failure of a test, etc.
- ``ratio_scale_interval`` – numerical data with a true zero point and equal intervals between values. E.g., height, weight, age, etc.
- ``numeric`` – general numerical data that can be treated as continuous values. E.g., temperature, income, etc.

Code examples can be found in the next section.

-------------------
Other parameters
-------------------

All parameters listed below are optional. If not provided, default values will be used. However, please note that
some of these configuration flags depend on one another. It will be clearly indicated in the description of each parameter.

- ``feature_weights (dict[int, float] | str | None)`` - Optional mapping of column indices (or names) to a float weight. If None or "uniform", all features will have equal weight of 1. Otherwise, the weights must be a dictionary mapping feature indices to weights, i.e. {0: 1.0, 1: 2.0}.
- ``data_type (type[np.integer | np.floating])`` - Optional flag to determine the data type that would be used during all calculation (and returned). If omitted, default to np.float32.
- ``scale_method (str)`` - Method used for scaling numerical features. Possible values are `range` and `iqr`. Defaults to `range`. If `range` is selected, numerical features will be scaled to the [0, 1] range. If `iqr` is selected, numerical features will be scaled using the interquartile range (IQR) method.
- ``scale_window (str | None)`` - Scaling window implementation flag for numeric or ratio features. Can be `None`, `kde` or `kNN`. Default is `None` if omitted.
- ``scale_window_type (str | None)`` - Type of scaling window to be used. Possible values are `None` or `silverman`. Default is `None`. This parameter is only relevant if `scale_window` is set to `kde`. In the future, more `kde` types might be added.
- ``missing_strategy (str)`` - Strategy for handling missing values. Unlike other libraries, Gower's metric can inherently handle missing values. This parameter allows you to specify how to treat them. Possible values are `ignore` (default), `max_dist` and `raise_error`. If set to `ignore`, missing values will be ignored in the distance calculation. If set to `max_dist`, missing values will be treated as having the maximum possible distance (1). If set to `raise_error`, an error will be raised if any missing values are encountered.
- ``categorical_ordinal_values_order (dict[int | str, list[str]] | None)`` - Required field when using `categorical_ordinal` feature type. It is a dictionary where keys are either column indices (int) or column names (str),
  and values are lists of strings representing the ordered categories for that feature. For example, if you have an ordinal feature representing education levels with categories "low", "medium", and "high", you would specify it as follows: 
  ``{0: ['low', 'medium', 'high']}``. If not provided, an error will be raised when `categorical_ordinal` feature type is used.
- ``categorical_ordinal_calculation_type (str)`` - Option to choose the calculation method for `categorical_ordinal` features. Possible values are `kaufman` and `podani`. Defaults to `kaufman`. More detailed description can be found `here <https://www.researchgate.net/publication/271789313_Extending_Gower%27s_General_Coefficient_of_Similarity_to_Ordinal_Characters>`_.
- ``k_neighbours (int | None)`` - Number of neighbors to consider when using kNN scaling window. Default is `None`, in which case the number of neighbors will be set to the square root of the number of points. This parameter is only relevant if `scale_window` is set to `kNN`. Can be 1 or higher.
- ``conditional_distances (bool)`` - If set to `True`, a two-step approach will be triggered. This acts as a blocking strategy inspired by Statistical Matching methods. First step involves calculating distances only for all binary and conditional categorical features.
  In the second step, ratio scale and numeric features are included in the calculation based on a threshold derived from the first step. Defaults to `False`.
  For more context, refer to `references year 2021 -> chapter 3 <https://arxiv.org/abs/2101.02481>`_.
- ``conditional_distances_threshold_coeff (int)`` - Threshold coefficient to be used when using conditional distances. Defaults to 1; it cannot be lower. Metric will be calculated as follows: if the distance calculated in the first step (using only binary and conditional categorical features)
  exceeds the threshold defined as ``conditional_distances_threshold_coeff * (p_cat / p)`` (where p_cat is the number of categorical and binary features, and p is the total number of features),
  then the final distance will be set to 1. Otherwise, the ratio scale and numeric features will be included in the distance calculation in the second step.