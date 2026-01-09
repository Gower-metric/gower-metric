====================
Basic examples
====================

Here we provide some basic code examples to demonstrate how to use the package with different types of data.

----------------------
Calculating distances
----------------------

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

    cfg = Config(
        feature_types=feature_types,
    )
    gower = Gower(cfg)
    gower.fit(data)

.. note::
   
   When passing numpy arrays with mixed data types, ensure that the array's dtype is set to *object* to accommodate different types of data.

We can also pass pandas DataFrames directly:

.. code-block:: python

    import pandas as pd

    from gower_metric import Gower
    from gower_metric.core.config import Config

    df = pd.DataFrame({
        "age": [23, 45, 23, 31],
        "gender": ["Female", "Male", "Female", "Male"],
        "income": [35000, 81000, 40000, 30000],
        "married": [0, 1, 1, 0],
        "infected": [1, 1, 0, 0],
    })

    feature_types = {
        "age": "ratio_scale_interval",
        "gender": "categorical_nominal",
        "income": "numeric",
        "married": "binary_symmetric",
        "infected": "binary_asymmetric",
    }
    cfg = Config(
        feature_types=feature_types,
        scale_method="iqr",
    )
    gower = Gower(cfg).fit(df)

-------------------------
Calculating similarities
-------------------------

Package provides an option to calculate similarities instead of distances.

.. code-block:: python

    import numpy as np

    from gower_metric import Gower
    from gower_metric.core.config import Config

    data = np.array([[1, 'a'], [2, 'b'], [3, 'a'], [4, 'c']], dtype=object)

    feature_types = {
        0: "ratio_scale_interval",
        1: "categorical_nominal",
    }

    cfg = Config(
        feature_types=feature_types,
    )
    gower = Gower(cfg).fit(data)
    
    similarity = gower.similarity(data[0], data[1])