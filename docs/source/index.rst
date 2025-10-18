=======================================
Welcome to gower-metric documentation!
=======================================

**Gower-metric** is a Python library for calculating distance or similarity for mixed-type variables 
derived as the complement of the Gower's similarity coefficient.

Main features include:

- Support for mixed data types (categorical, numerical, binary)
- Podani's support
- Efficient computation using NumPy framework
- Numerical friendly thanks to transform call and joblib support
- SciPy sparse matrix support
- Easy integration with pandas DataFrames
- Customizable weighting for different variable types
- Friendly MIT License

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
   Currently, the package is optimized for clarity and correctness, but we plan to enhance its performance using lower-level languages (Cython) or Numba for JIT compilation.
- GPU parallelization support. 
   Leveraging GPU capabilities can significantly speed up computations, especially for large datasets. We plan to explore libraries like CuPy or RAPIDS for this purpose.
- Continous optimization support. 
   Here is the problem. For now, continous variable optimization is not being done iteratively. It compares calculated results with set threshold only once.
   It is possible to make it iterative, so the algorithm will keep changing the threshold until it reaches desired state. However,
   this will require more numerical speed optimizations.
- Advanced weights optimization techniques. 
   Currently, the package supports basic weight handling methodology. We plan to implement more advanced techniques, such as cophenetic correlation optimization and index of agreement.
- Kernel density estimator bandwidth selection methods. 
   Currently, the package supports Silverman's rule of thumb for bandwidth selection. We plan to add more methods, such as Scott's rule.

If you have any suggestions or would like to contribute to these improvements, please feel free to reach out.

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

.. toctree::
   :maxdepth: 1
   :hidden:

   compatibility

.. toctree::
   :maxdepth: 1
   :hidden:

   metric_description