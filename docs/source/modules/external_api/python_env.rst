.. _python_environment_external_api:

===================
Python environment
===================

Developer's team provide compatibility with libraries such as `scipy <https://scipy.org/>`_ and `scikit-learn <https://scikit-learn.org/stable/>`_.

-----------------------------
How to make Gower compatible
-----------------------------

User can not call our metric directly in those libraries, due to the fact that those libraries expect numerical input data. 
Therefore, user needs to first transform the original data into its numerical representation using the ``transform`` or ``fit_transform`` methods.
More information with examples can be found in following subsections.

.. note::

    We are aware that calling external libraries can be slower in performance. Please do wait for future updates, where we will provide
    more optimized kernels.

Examples can be found in the ``examples`` folder of the repository.

.. toctree::
   :maxdepth: 1

   scipy/scipy
   scikit_learn/scikit_learn