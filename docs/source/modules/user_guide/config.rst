.. _configuration_class:

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
- ``data_type (type[np.floating] | None)`` - Optional flag to determine the data type used during all calculations (and returned). If omitted, defaults to np.float32.
- ``scale_method (str)`` - Method used for scaling numerical features. Possible values are `range` and `iqr`. Defaults to `range`. If `range` is selected, numerical features will be scaled to the [0, 1] range. If `iqr` is selected, numerical features will be scaled using the interquartile range (IQR) method.
- ``scale_window (str | None)`` - Scaling window implementation flag for numeric or ratio features. Can be `None`, `kde` or `kNN`. Default is `None` if omitted.
- ``scale_window_type (str | None)`` - Type of scaling window to be used. Possible values are `None` or `silverman`. Default is `None`. This parameter is only relevant if `scale_window` is set to `kde`. In the future, more `kde` types might be added.
- ``missing_strategy (str)`` - Strategy for handling missing values. Unlike other libraries, Gower's metric can inherently handle missing values. This parameter allows you to specify how to treat them. Possible values are `ignore` (default), `max_dist` and `raise_error`. If set to `ignore`, missing values will be ignored in the distance calculation. If set to `max_dist`, missing values will be treated as having the maximum possible distance (1). If set to `raise_error`, an error will be raised if any missing values are encountered.
- ``categorical_ordinal_values_order (dict[int | str, list[str]] | None)`` - Required field when using `categorical_ordinal` feature type. It is a dictionary where keys are either column indices (int) or column names (str),
  and values are lists of strings representing the ordered categories for that feature. For example, if you have an ordinal feature representing education levels with categories "low", "medium", and "high", you would specify it as follows:
  ``{0: ['low', 'medium', 'high']}``. If not provided, an error will be raised when `categorical_ordinal` feature type is used.
- ``categorical_ordinal_calculation_type (str)`` - Option to choose the calculation method for `categorical_ordinal` features. Possible values are `kaufman` and `podani`. Defaults to `kaufman`. More detailed description can be found `here <https://www.researchgate.net/publication/271789313_Extending_Gower%27s_General_Coefficient_of_Similarity_to_Ordinal_Characters>`_.
- ``k_neighbors (int | None)`` - Number of neighbors to consider when using kNN scaling window. Default is `None`, in which case the number of neighbors will be set to the square root of the number of points. This parameter is only relevant if `scale_window` is set to `kNN`. Can be 1 or higher.
- ``conditional_distances (bool)`` - If set to `True`, a two-step approach will be triggered. This acts as a blocking strategy inspired by Statistical Matching methods. First step involves calculating distances only for all binary and conditional categorical features.
  In the second step, ratio scale and numeric features are included in the calculation based on a threshold derived from the first step. Defaults to `False`.
  For more context, refer to `references year 2021 -> chapter 3 <https://arxiv.org/abs/2101.02481>`_.
- ``conditional_distances_threshold_coeff (int)`` - Threshold coefficient to be used when using conditional distances. Defaults to 1; it cannot be lower. Metric will be calculated as follows: if the distance calculated in the first step (using only binary and conditional categorical features)
  exceeds the threshold defined as ``conditional_distances_threshold_coeff * (p_cat / p)`` (where p_cat is the number of categorical and binary features, and p is the total number of features),
  then the final distance will be set to 1. Otherwise, the ratio scale and numeric features will be included in the distance calculation in the second step.
- ``out_of_range (str)`` - Strategy for handling numeric and ratio-scale values that fall outside the range observed during ``fit()``. Possible values are ``"clip"`` (silently clip normalized distances to [0, 1]), ``"warning"`` (default, emit a ``UserWarning`` listing offending columns and clip), or ``"error"`` (raise a ``ValueError``). This applies to both ``transform()`` and pairwise distance calculations (``__call__``, ``matrix``).

----------------------------
Handling unseen values
----------------------------

When working with a train/test split, it's common for the test set to contain values that were never seen during training.
The ``handle_unseen_*`` parameters let you control what happens in that situation. All of them accept the same three strategies:

- ``"error"`` (default) – raise a ``ValueError`` when an unseen value is encountered. This is the safest option and forces you to deal with the issue explicitly.
- ``"warning"`` – silently map the unseen value to ``NaN`` (i.e. treat it as missing), but emit a ``UserWarning`` so you know it happened.
- ``"missing"`` – silently map it to ``NaN`` without any warning. Handy when you already know unseen values are expected and acceptable.

Each feature type has its own toggle:

- ``handle_unseen_binary_asymmetric (str)`` - Strategy for handling unseen values in binary asymmetric features. Defaults to ``"error"``.
- ``handle_unseen_binary_symmetric (str)`` - Strategy for handling unseen values in binary symmetric features. Defaults to ``"error"``.
- ``handle_unseen_categorical_nominal (str)`` - Strategy for handling unseen values in categorical nominal features. Defaults to ``"error"``.
- ``handle_unseen_categorical_ordinal (str)`` - Strategy for handling unseen values in categorical ordinal features. Defaults to ``"error"``.

.. code-block:: python

   from gower_metric import Config, Gower

   cfg = Config(
       feature_types={0: "binary_asymmetric", 1: "categorical_nominal"},
       handle_unseen_binary_asymmetric="warning",
       handle_unseen_categorical_nominal="missing",
   )

----------------------------
Binary value ordering
----------------------------

By default, binary features auto-detect their two possible values from the training data and sort them alphabetically.
If your domain requires precise control over which value maps to 0 and which maps to 1, you can specify an explicit ordering.
This is especially useful when the training set only contains one of the two expected values (a *degenerate fit*).

- ``binary_asymmetric_value_order (dict[int | str, list[Any]] | None)`` - Optional explicit ordering for binary asymmetric features. Each entry maps a column index (or name) to a list of exactly two values, where the first value becomes 0 and the second becomes 1. Defaults to ``None`` (auto-detect).
- ``binary_symmetric_value_order (dict[int | str, list[Any]] | None)`` - Same as above, but for binary symmetric features. Defaults to ``None`` (auto-detect).

.. code-block:: python

   cfg = Config(
       feature_types={0: "binary_asymmetric"},
       binary_asymmetric_value_order={0: ["No", "Yes"]},
   )

-------------------------------
Out-of-range numeric values
-------------------------------

When you fit on a training set and then transform or compute distances on new data, numeric and ratio-scale
features may contain values outside the range seen during ``fit()``. By default, Gower emits a warning and
clips normalized distances to [0, 1] — consistent with the original Gower (1971) property that
:math:`d_{ij}^k \in [0, 1]`.

The ``out_of_range`` parameter lets you control this behavior:

- ``"warning"`` (default) – emit a ``UserWarning`` listing each offending column with its actual and fitted range, then clip.
- ``"clip"`` – silently clip without any notification.
- ``"error"`` – raise a ``ValueError`` so you can investigate before proceeding.

.. code-block:: python

   import numpy as np
   from gower_metric import Config, Gower

   X_train = np.array([[1.0], [5.0]], dtype=object)
   X_test = np.array([[10.0]], dtype=object)  # outside [1, 5]

   # Default: emits a warning
   cfg = Config(
       feature_types={0: "numeric"},
   )
   gower = Gower(cfg).fit(X_train)
   gower.transform(X_test)  # UserWarning: Out-of-range values detected ...

   # Strict: raise an error instead
   cfg_strict = Config(
       feature_types={0: "numeric"},
       out_of_range="error",
   )
   gower_strict = Gower(cfg_strict).fit(X_train)
   gower_strict.transform(X_test)  # raises ValueError

   # Silent: no warning, just clip
   cfg_silent = Config(
       feature_types={0: "numeric"},
       out_of_range="clip",
   )
   gower_silent = Gower(cfg_silent).fit(X_train)
   gower_silent.transform(X_test)  # no output, distances clipped to [0, 1]