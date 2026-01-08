===============
Matrix support
===============

We also provide endpoint support for calculating various types of matrices. We use `joblib <https://github.com/joblib/joblib>`_ library to parallelize computations and speed up the whole process.
By default, *backend* is set to *loky*.

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

   distance_matrix = gower.matrix(data)
   similarity_matrix = gower.matrix(data, matrix_type='similarity')

Endpoint creates a square matrix where each element (i, j) represents the distance or similarity between data points i and j. You can specify the type of matrix you want to compute using the ``matrix_type`` parameter, which can be either *distance* or *similarity*. By default, it computes the distance matrix.

----------------------
Sparse matrix support
----------------------

On top of that, user can also compute sparse matrices using SciPy's sparse matrix format. This is particularly useful when dealing with large datasets where many distances or similarities are zero, allowing for more efficient storage and computation.

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

   sparse_distance_matrix = gower.matrix(data, convert_to_sparse=True, sparse_type="csc")
   sparse_similarity_matrix = gower.matrix(data, matrix_type='similarity', convert_to_sparse=True, sparse_type="coo")

Here, the ``convert_to_sparse`` parameter is set to *True* to indicate that we want the output in a sparse format. The ``sparse_type`` parameter allows you to choose the specific type of sparse matrix representation, such as *csc* (Compressed Sparse Column), *csr* (Compressed Sparse Row) , or *coo* (Coordinate List).
By default, ``n_jobs`` is set to -1, which means that all available CPU cores will be used for parallel computation. You can adjust this parameter based on your system's capabilities and the size of your dataset. Default sparse type is *csr*.

----------------------
Custom created matrix
----------------------

User can also create matrix *by hand* and fill it with values.

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

   matrix = np.zeros((data.shape[0], data.shape[0]))
   
   for i in range(data.shape[0]):
       for j in range(data.shape[0]):
           matrix[i, j] = gower(data[i], data[j])

This approach allows for more flexibility, as you can define your own logic for populating the matrix based on specific criteria or conditions.
It could be improved by *joblib* parallelization if needed.