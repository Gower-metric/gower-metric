================
R environment
================

In this section we dive deeper into numerical comparison between our implementation and R's enviroment. We used two main R packages:

- **daisy**: This package provides the `daisy <https://www.rdocumentation.org/packages/cluster/versions/2.1.8.1/topics/daisy>`_ function, which computes dissimilarities for mixed-type data, including Gower's distance.
- **cran**: This `package <https://cran.r-universe.dev/gower/doc/manual.html>`_ provides a dedicated implementation of Gower's metric with openMP support.

In following subsections, we present the results and speed comparisons.

.. toctree::
   :maxdepth: 1

   daisy/r_daisy
   cran/r_cran