Welcome to gower-metric documentation!
======================================

**Gower-metric** is a Python library for calculating distance for mixed-type variables 
derived as the complement of the Gower's similarity coefficient.

Main features include:

- Support for mixed data types (categorical, numerical, ordinal, binary)
- Podani's support
- Efficient computation using NumPy framework
- Numerical friendly thanks to transform call and joblib support
- SciPy sparse matrix support
- Easy integration with pandas DataFrames
- Customizable weighting for different variable types
- MIT License

.. note::

   This project is under active development. If you would like to contribute,
   or if you find any issues, please visit the main [repository]() and 
   submit a pull request or open an issue.

-----------
Quick start
-----------

In order to import class, which calculate Gower's metric, you need to import it as follows:

.. code-block:: python

   from gower_metric import Gower

After that, we have to initialize the features type dictionary:

.. code-block:: python

   data = [[1, 'a', 3.5], [2, 'b', 4.0], [3, 'a', 2.5], [4, 'c', 5.0]]

   feature_types = {
      0: "ratio_scale_interval",
      1: "categorical_nominal",
      2: "ratio_scale_interval"
   }

   gower = Gower(feature_types=feature_types)

Finally, we can fit our data and calculate Gower's distance over first and second rows:

.. code-block:: python

   gower.fit(data)
   distance = gower(data[0], data[1])

.. toctree::
   :maxdepth: 1

   installation

.. toctree::
   :maxdepth: 1

   user_guide

.. toctree::
   :maxdepth: 1

   api_reference
