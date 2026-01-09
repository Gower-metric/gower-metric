==================
Advanced examples
==================

In this section we will show how to use weighting of features, categorical ordinal data type and Podani's optimization.

----------------
Passing weights
----------------

You can pass weights to features when calculating distances or similarities. This allows you to give more importance to certain features over others.

.. code-block:: python

    import numpy as np

    from gower_metric import Gower
    from gower_metric.core.config import Config

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

    from gower_metric import Gower
    from gower_metric.core.config import Config

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

    from gower_metric import Gower

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
        categorical_ordinal_calculation_type="podani",
        scale_method="iqr",
        missing_strategy="max_dist",
        scale_window="kde",
        scale_window_type="silverman",
        conditional_distances=True
    )
    gower = Gower(cfg)
    gower.fit(data)