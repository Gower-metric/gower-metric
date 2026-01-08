======
SciPy
======

We assume that you called the ``transform`` method and imported scipy library.

-----------------
Spatial distance
-----------------

.. code-block:: python

    from scipy.spatial.distance import pdist, squareform
    from gower_metric import Gower
    from gower_metric.core.config import Config

    # fit Gower model first
    cfg = Config(
        feature_types=feature_types,
    )
    gower = Gower(cfg).fit(df)

    # calculate the numerical representation of original data
    df_num = gower.transform(df)

    # calculate pairwise distance using scipy
    array_scipy = pdist(df_num, metric=gower)

    # convert to square matrix form
    matrix_scipy = squareform(array_scipy)

There are no changes required, as scipy supports custom NaN handling in its distance functions, which, by definition,
is one of the features of Gower's metric.

---------------------------
Cophenet cluster hierarchy
---------------------------

.. code-block:: python

    from scipy.cluster.hierarchy import cophenet, single
    from scipy.spatial.distance import pdist, squareform
    from gower_metric import Gower
    from gower_metric.core.config import Config

    cfg = Config(
        feature_types=feature_types,
    )
    gower = Gower(cfg).fit(df)
    df_num = gower.transform(df)

    Z = single(pdist(df_num, metric=gower))
    c, _ = cophenet(Z, pdist(df_num, metric=gower))

    # print the results if needed
    print(f"{squareform(cophenet(Z))}")

.. automodule::
    :members:
    :undoc-members:
    :show-inheritance: