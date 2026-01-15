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

---------------------------------------
Do we validate if data is transformed?
---------------------------------------

Yes, and no. Here is the catch, we validate if data is transformed only when dealing with pandas DataFrames. Why? Because they support
metadata field within the DataFrame object, unlike numpy arrays (well, there has been such feature but is no longer supported).

In order to maintain compatibility with external libraries API, there are no more advanced checks of non-dataframe data. Please be aware of that.
``Thus user should not call Gower instance on original data after transformation!``