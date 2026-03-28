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

   from gower_metric import Config, Gower

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

   from gower_metric import Config, Gower

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

-----------------------------------
Out-of-range values in transform
-----------------------------------

When numeric or ratio-scale features in the transform (or test) data contain values outside the range
observed during ``fit()``, the ``out_of_range`` config parameter controls the behavior.
By default it is set to ``"warning"`` — a ``UserWarning`` is emitted listing each column where values
exceed the fitted range, and the resulting normalized distances are clipped to [0, 1].

.. code-block:: python

   import numpy as np
   from gower_metric import Config, Gower

   X_train = np.array([[1.0], [5.0]], dtype=object)
   X_test = np.array([[10.0]], dtype=object)  # outside [1, 5]

   cfg = Config(
       feature_types={0: "numeric"},
       out_of_range="error",  # raise instead of warn
   )
   gower = Gower(cfg).fit(X_train)
   gower.transform(X_test)  # raises ValueError

For more details and all available strategies, see :ref:`Configuration Class <configuration_class>`.

---------------------------
Unseen values in transform
---------------------------

In a typical ML workflow, you'll fit on a training set and transform a separate test set. If the test set contains
values that weren't present during training, Gower's ``transform`` will behave differently depending on the
``handle_unseen_*`` flags you set in your ``Config``.

By default, all feature types use ``"error"``, which raises a ``ValueError`` the moment it hits something unknown.
This is intentional – it forces you to make a conscious decision about how to handle these cases.

.. code-block:: python

   import numpy as np
   from gower_metric import Config, Gower

   X_train = np.array([["A"], ["B"]], dtype=object)
   X_test = np.array([["C"]], dtype=object)  # "C" was never in training

   cfg = Config(
      feature_types={0: "categorical_nominal"},
      handle_unseen_categorical_nominal="error",  # default
   )
   gower = Gower(cfg).fit(X_train)
   gower.transform(X_test)  # raises ValueError

If you'd rather treat unseen values as missing data (``NaN``), switch to ``"missing"`` or ``"warning"``:

.. code-block:: python

   cfg = Config(
      feature_types={0: "categorical_nominal"},
      handle_unseen_categorical_nominal="missing",
   )
   gower = Gower(cfg).fit(X_train)
   result = gower.transform(X_test)   # "C" -> NaN, no error

The same applies to binary features. If you know the expected binary values up front, combining
``binary_asymmetric_value_order`` with ``handle_unseen_binary_asymmetric="missing"`` gives you
full control over how unseen values are mapped:

.. code-block:: python

   cfg = Config(
      feature_types={0: "binary_asymmetric"},
      binary_asymmetric_value_order={0: ["No", "Yes"]},
      handle_unseen_binary_asymmetric="missing",
   )
   # Training set only has "No" – "Yes" is still recognized thanks to explicit order
   gower = Gower(cfg).fit(np.array([["No"], ["No"]], dtype=object))
   result = gower.transform(np.array([["Yes"]], dtype=object))   # -> 1.0

For more details on available strategies, see the :ref:`Configuration Class <configuration_class>` section.