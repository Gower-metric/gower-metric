Welcome to gower-metric documentation!
======================================

**Gower-metric** is a Python library for calculating distance or similarity for mixed-type variables 
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

Installation, user guide, and API reference can be found in the respective sections. We also provide separate section 
for describing how Gower's metric formula works.

--------------------
Future improvements
--------------------

We understand that there are still some features missing in the current version of the package. Here are some of the planned improvements for future releases:

- Numerical pipeline optimizations. 
   Currently, the package is optimized for clarity and correctness, but we plan to enhance its performance using lower-level languages (Cython, Numba) and parallel processing techniques.
- Continous optimization support. 
   Here is the problem. For now, continous variable optimization is not being done iteratively. It compares calculated results with set threshold only once.
   It is possible to make it iterative, so the algorithm will keep changing the threshold until it reaches desired state. However,
   this will require more numerical speed optimizations.

.. toctree::
   :maxdepth: 1
   :hidden:

   installation

.. toctree::
   :maxdepth: 1
   :hidden:

   user_guide

.. toctree::
   :maxdepth: 1
   :hidden:

   api_reference
