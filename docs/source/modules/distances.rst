=========
Distances
=========

.. automodule:: gower_metric.distances
   :members:
   :undoc-members:
   :show-inheritance:

.. tip::

   To calculate the pairwise distances for the entire dataset, you can do it manually or
   use an auxiliary function, like: `scipy.spatial.distance.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_
   or `sklearn.metrics.pairwise_distances <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html>`_.

.. toctree::
   :maxdepth: 1
   
   distances/binary_asymmetric
   distances/binary_symmetric
   distances/categorical_nominal
   distances/categorical_ordinal
   distances/numeric_interval
   distances/ratio_scale_interval