=================
Transform method
=================

The ``transform`` method allows user to convert original dataframe into its numerical representation. Thanks to that, 
we can use our metric with external libraries such as scipy or scikit-learn. More on that in :ref:`Python External Api Compatibility <python_environment_external_api>` section.

----------
Transform
----------

It is done by calling the ``transform`` method on a fitted Gower instance.

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

   transformed_data = gower.transform(data)

All numerical, ratio scale interval features are intact. Binary features are represented as 0 and 1, depending on the value.
Categorical nominal and ordinal features are encoded using ordinal encoding from scikit-learn.

--------------
Fit transform
--------------

For convenience, Gower also implements ``fit_transform`` method, which combines ``fit`` and ``transform`` in one call.

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
   gower = Gower(cfg)

   transformed_data = gower.fit_transform(data)

.. warning::

    Under the hood, the ``fit`` method learns the mapping of categorical ordinal values to their numerical representation.
    Therefore, calling ``transform`` on the same data before and after re-fitting the instance may result in different numerical representations
    or even NaN values. The same applies to the ``fit_transform`` method. If your data does not contain any categorical ordinal features,
    this warning may not apply (we have not yet implemented tests for this scenario).