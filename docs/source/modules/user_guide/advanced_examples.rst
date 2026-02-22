==================
Advanced examples
==================

In this section we will show how to use weighting of features, categorical ordinal data type, Podani's optimization,
and practical train/test split workflows with full ``Config`` usage.

----------------
Passing weights
----------------

You can pass weights to features when calculating distances or similarities. This allows you to give more importance to certain features over others.

.. code-block:: python

    import numpy as np

    from gower_metric import Config, Gower

    data = np.array([[1, 'a', 3.5], [2, 'b', 4.0], [3, 'a', 2.5], [4, 'c', 5.0]], dtype=object)

    feature_types = {
        0: "ratio_scale_interval",
        1: "categorical_nominal",
        2: "ratio_scale_interval"
    }

    weights = {
        0: 0.5,
        1: 2.0,
        2: 1.0
    }

    cfg = Config(
        feature_types=feature_types,
        feature_weights=weights,
    )
    gower = Gower(cfg).fit(data)

----------------------------
Categorical ordinal example
----------------------------

Here things get a bit more tricky. When dealing with categorical ordinal data, we need to provide an additional mapping that defines the order of the categories.

.. code-block:: python

    import numpy as np

    from gower_metric import Config, Gower

    data = np.array([
        [1, 'low', 3.5],
        [2, 'medium', 4.0],
        [3, 'high', 2.5],
        [4, 'medium', 5.0]
    ], dtype=object)

    feature_types = {
        0: "ratio_scale_interval",
        1: "categorical_ordinal",
        2: "ratio_scale_interval"
    }

    ordinal_mappings = {
        1: ['low', 'medium', 'high']
    }

    cfg = Config(
        feature_types=feature_types,
        categorical_ordinal_values_order=ordinal_mappings,
    )
    gower = Gower(cfg)
    gower.fit(data)

-------------------------
More class functionality
-------------------------

On top of the examples before, we can also play with other class functionalities, mostly for numerical data but not only.

.. code-block:: python

    import numpy as np

    from gower_metric import Config, Gower

    DTYPE = np.float64

    data = np.array([
        [1, 'low', 3.5],
        [2, 'medium', 4.0],
        [3, 'high', 2.5],
        [4, 'medium', 5.0]
    ], dtype=object)

    feature_types = {
        0: "ratio_scale_interval",
        1: "categorical_ordinal",
        2: "ratio_scale_interval"
    }

    ordinal_mappings = {
        1: ['low', 'medium', 'high']
    }

    cfg = Config(
        feature_types=feature_types,
        data_type=DTYPE,
        categorical_ordinal_values_order=ordinal_mappings,
        categorical_ordinal_calculation_type="podani",
        scale_method="iqr",
        missing_strategy="max_dist",
        scale_window="kde",
        scale_window_type="silverman",
        conditional_distances=True,
    )
    gower = Gower(cfg)
    gower.fit(data)

-------------------------------------
Train / test split (without stratify)
-------------------------------------

When working with a train/test split, you fit on the training set and transform both sets separately.
Since the test set may contain values never seen during training, you should decide on a ``handle_unseen_*``
strategy up front.

.. code-block:: python

    import numpy as np
    from sklearn.model_selection import train_test_split

    from gower_metric import Config, Gower

    data = np.array([
        [25, "low",    "car",   True,  0],
        [30, "medium", "bus",   False, 1],
        [35, "high",   "car",   True,  0],
        [40, "low",    "train", False, 1],
        [28, "medium", "car",   True,  0],
        [50, "high",   "bus",   False, 1],
        [22, "low",    "train", True,  0],
        [45, "medium", "car",   False, 1],
    ], dtype=object)

    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.25, random_state=42,
    )

    cfg = Config(
        feature_types={
            0: "ratio_scale_interval",
            1: "categorical_ordinal",
            2: "categorical_nominal",
            3: "binary_symmetric",
            4: "binary_asymmetric",
        },
        categorical_ordinal_values_order={1: ["low", "medium", "high"]},
        # Unseen values in the test set will be silently mapped to NaN
        handle_unseen_categorical_nominal="missing",
        handle_unseen_categorical_ordinal="missing",
        handle_unseen_binary_asymmetric="missing",
        handle_unseen_binary_symmetric="missing",
    )

    gower = Gower(cfg).fit(X_train)

    X_train_t = gower.transform(X_train)
    X_test_t = gower.transform(X_test)

    # Both arrays are now fully numeric – ready for scikit-learn
    print(X_train_t)
    print(X_test_t)

.. tip::

   Using ``"missing"`` is the easiest option for exploratory work. For production pipelines,
   consider ``"error"`` (the default) so you're immediately alerted when something unexpected
   shows up in the data.

--------------------------------
Stratified split with KNeighbors
--------------------------------

Stratified splitting preserves class proportions, which is important for imbalanced datasets.
Here's a complete example combining all Config options with scikit-learn's KNN.

.. code-block:: python

    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    from gower_metric import Config, Gower

    df = pd.DataFrame({
        "age":      [25, 30, 35, 40, 28, 50, 22, 45, 33, 38],
        "level":    ["low", "medium", "high", "low", "medium",
                     "high", "low", "medium", "high", "low"],
        "vehicle":  ["car", "bus", "car", "train", "car",
                     "bus", "train", "car", "bus", "train"],
        "married":  [True, False, True, False, True,
                     False, True, False, True, False],
        "infected": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3, random_state=42, stratify=y,
    )

    cfg = Config(
        feature_types={
            "age":      "ratio_scale_interval",
            "level":    "categorical_ordinal",
            "vehicle":  "categorical_nominal",
            "married":  "binary_symmetric",
            "infected": "binary_asymmetric",
        },
        categorical_ordinal_values_order={"level": ["low", "medium", "high"]},
        categorical_ordinal_calculation_type="podani",
        handle_unseen_categorical_nominal="warning",
        handle_unseen_categorical_ordinal="warning",
        handle_unseen_binary_asymmetric="warning",
        handle_unseen_binary_symmetric="warning",
    )

    gower = Gower(cfg).fit(X_train)
    X_train_t = gower.transform(X_train)
    X_test_t = gower.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3, metric=gower, n_jobs=-1)
    knn.fit(X_train_t, y_train)
    y_pred = knn.predict(X_test_t)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

.. note::

   The ``stratify=y`` argument ensures that the class distribution in training and test sets
   mirrors the original dataset. This is particularly important when working with binary asymmetric
   features where the "positive" class may be rare.

-------------------------------------------
Degenerate fit with explicit value ordering
-------------------------------------------

Sometimes the training set only contains one of the two possible binary values – a *degenerate fit*.
Without explicit ordering, the unseen value would be treated as unknown. With ``binary_asymmetric_value_order``
or ``binary_symmetric_value_order``, Gower knows both expected values upfront:

.. code-block:: python

    import numpy as np

    from gower_metric import Config, Gower

    # Training set: only "No" is present
    X_train = np.array([["No"], ["No"], ["No"]], dtype=object)
    # Test set: "Yes" appears for the first time
    X_test = np.array([["Yes"], ["No"]], dtype=object)

    cfg = Config(
        feature_types={0: "binary_asymmetric"},
        binary_asymmetric_value_order={0: ["No", "Yes"]},  # No -> 0.0, Yes -> 1.0
    )

    gower = Gower(cfg).fit(X_train)
    result = gower.transform(X_test)

    # "Yes" is correctly mapped to 1.0 despite never appearing in training
    print(result)  # [[1.0], [0.0]]

.. tip::

   Explicit value ordering is especially useful in medical or survey data where you *know* both
   possible outcomes (e.g. positive/negative, yes/no) but your training sample might be skewed
   towards one of them.