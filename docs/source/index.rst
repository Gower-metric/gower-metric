=======================================
Welcome to gower-metric documentation!
=======================================

**Gower-metric** is a Python library for calculating distance or similarity for mixed-type variables
derived as the complement of the Gower's similarity coefficient.

Main features include:

- Support for mixed data types (categorical, numerical, binary)
- Easy integration with pandas DataFrames and NumPy arrays
- Customizable weighting for different variable types
- SciPy sparse matrix support
- Podani's support
- Friendly MIT License

.. note::

   This project is under active development. If you would like to contribute,
   or if you find any issues, please visit the main `repository <https://github.com/Gower-metric/gower-metric>`_ and
   submit a pull request or open an issue.

Installation, user guide, and API reference can be found in the respective sections. We also provide separate section
for describing how Gower's metric formula works and how it can be used with external libraries.

------------------------
Documentation structure
------------------------

The documentation is organized into several sections to help you get started and make the most out of the gower-metric package:

- **Installation**: Instructions on how to install the package, how to contribute to its development and how to set up the development environment.
- **User Guide**: A comprehensive guide on how to use the package, including simple and more advanced examples.
- **API Reference**: Detailed documentation of all classes, functions, and methods available in the package.
- **External API Compatibility**: Information about the package's compatibility with external libraries and frameworks, such as SciPy and Scikit-learn. In addition, we also provide comparison with R implementation and benchmark subsection.
- **Metric Description**: An in-depth explanation of Gower's metric, including the mathematical formulation and how it handles different variable types.

--------------------
Future improvements
--------------------

We understand that there are still some features missing in the current version of the package. Here are some of the planned improvements for future releases:

- Numerical pipeline optimizations.
   Currently, the package is optimized for clarity and correctness, but we plan to enhance its performance using lower-level languages (Cython) or Numba for JIT compilation.
- GPU parallelization support.
   Leveraging GPU capabilities can significantly speed up computations, especially for large datasets. We plan to explore libraries like CuPy or RAPIDS for this purpose.
- Conditional distances support.
   Here is the problem. For now, conditional distances optimization is not being done iteratively. It compares calculated results with set, by user, threshold.
   That's it. It is possible to make it iterative, so the algorithm will keep changing the threshold until it reaches desired state. However,
   this will require more numerical speed optimizations. More on that `here <https://arxiv.org/abs/2101.02481>`_ (chapter 3).
- Advanced weights optimization techniques.
   Currently, the package supports basic weight handling methodology. We plan to implement more advanced handling, saving and loading weights to/from files.
- Kernel density estimator bandwidth selection methods.
   Currently, the package supports Silverman's rule of thumb for bandwidth selection. We plan to add more methods, such as Scott's rule.

If you have any suggestions or would like to contribute to these improvements, please feel free to reach out.

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   user_guide
   api_reference
   compatibility
   metric_description