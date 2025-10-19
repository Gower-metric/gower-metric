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

Import the module
------------------

In order to import class module, you might import it as follows:

.. code-block:: python

   from gower_metric import Gower

Using `fit` method
-------------------

To calculate Gower's distance, you first need to initialize the feature types dictionary and fit the model to your data.
It is only required variable to call class Gower. Any possible errors might arise from incorrect feature types dictionary.
Let's assume we imported the class as shown above and we have the following data:

.. code-block:: python

   import numpy as np
   
   from gower_metric import Gower
   
   data = np.array([[1], [4], [7]], dtype=float)
   f_types = {0: "ratio_scale_interval"}
   
   gower = Gower(f_types).fit(data)

As you can see, we initialized the feature types dictionary and created an instance of Gower class. After that, we called the `fit` method with our data.
We can easly use pd.DataFrame as input data as well.

.. automodule:: 
    :members:
    :undoc-members:
    :show-inheritance: