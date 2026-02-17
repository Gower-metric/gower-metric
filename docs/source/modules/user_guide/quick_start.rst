============
Quick Start
============

Package provides functionality to calculate Gower's distance or similarity between two data points.
It supports different types of features, as follows:

-   categorical_nominal
-   categorical_ordinal
-   binary_symmetric
-   binary_asymmetric
-   ratio_scale_interval
-   numeric

.. important::
   
   It is crucial to not be mistaken here! The keys of the dictionary must correspond to the indices of the columns in your dataset,
   and the values must accurately represent the type of data in each column. This ensures that the Gower's metric is calculated 
   correctly based on the nature of each feature.

------------------
Import the module
------------------

In order to import class module, you might import it as follows:

.. code-block:: python

   from gower_metric import Config, Gower

---------------------
Using ``fit`` method
---------------------

To calculate Gower's distance, you first need to initialize the feature types dictionary and fit the model to your data.
It is only required variable to call class Gower. Any possible errors might arise from incorrect feature types dictionary.
Let's assume we imported the class as shown above and we have the following data:

.. code-block:: python

   import numpy as np
   
   from gower_metric import Config, Gower
   
   data = np.array([[1], [4], [7]], dtype=float)
   f_types = {0: "ratio_scale_interval"}
   
   cfg = Config(
      feature_types=f_types,
   )
   gower = Gower(cfg).fit(data)

As you can see, we initialized the feature types dictionary and created an instance of Gower class. After that, we called the ``fit`` method with our data.
We can easly use pd.DataFrame as input data as well.

----------------------
What is Config class?
----------------------

Config class is being used to pass any configuration parameters into Gower instance. The only one required field is ``feature_types``, which 
indicates the type of each feature in the dataset. Other parameters are optional. Long story short, ``feature_types`` is a dictionary
where keys are either ``int`` or ``str`` (column indices or names respectively) and values are ``str`` type. Any missnaming or
incorrect specification will lead to error raise or calculation missmatch. Example below.

.. code-block:: python

   f_types = {
      0: "categorical_nominal",
      1: "binary_symmetric",
      2: "ratio_scale_interval",
   }

   cfg = Config(
      feature_types=f_types,
   )

For more detailed specification of Config class, please refer to next subsection. If you would like to see mathematical explanation
regarding Gower's metric, you can check the :ref:`metric description <metric_description>` page.

--------------------------------
Should I do data preprocessing?
--------------------------------

Yes, you should. For instance, missing values should be anything detectable by python's ``math.isnan`` function, panda's ``pd.isna`` or numpy's ``np.isnan``.
For example, if you have some ``?`` values in your dataset, you should replace them with ``np.nan``.

.. toctree::
   :maxdepth: 1
   :hidden:

   config