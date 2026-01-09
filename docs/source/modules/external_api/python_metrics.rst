===================
Metrics Comparison
===================

In this section we provide detailed comparison of Gower's metric algorithm with metrics such as:

- `kNN <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_
- `Hierarchical Clustering <https://docs.scipy.org/doc/scipy-1.15.0/reference/cluster.hierarchy.html>`_
- `HDBSCAN <https://pypi.org/project/hdbscan/>`_

As background for hyperparameters improvement, we have used `Optuna <https://optuna.org>`_ framework. The results are shown in the following tables.

----
KNN
----

Adult Dataset

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - **Metric**
     - **KNN Score**
     - **F1**
   * - Gower
     - 0.7840
     - 0.7759
   * - Euclidean
     - 0.7780
     - 0.7432
   * - Cosine
     - 0.7950
     - 0.7886
   * - Minkowski
     - 0.7490
     - **0.7050**
   * - Dice
     - **0.7270**
     - 0.7418
   * - Jaccard
     - 0.7450
     - 0.7620


Car Insurance Claim Dataset

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - **Metric**
     - **KNN Score**
     - **F1**
   * - Gower
     - 0.7420
     - 0.7278
   * - Euclidean
     - 0.7060
     - 0.6675
   * - Cosine
     - 0.7000
     - 0.6643
   * - Minkowski
     - 0.6980
     - **0.6586**
   * - Dice
     - 0.7270
     - 0.6999
   * - Jaccard
     - **0.6940**
     - 0.6922


Diabetes Dataset


.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - **Metric**
     - **KNN Score**
     - **F1**
   * - Gower
     - 0.6883
     - 0.6272
   * - Euclidean
     - 0.6883
     - 0.6839
   * - Cosine
     - 0.7208
     - 0.7202
   * - Minkowski
     - 0.7727
     - 0.7733
   * - Dice
     - 0.6558
     - 0.5195
   * - Jaccard
     - **0.6039**
     - **0.4548**


------------------------
Hierarchical clustering
------------------------

Adult Dataset

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - **Metric**
     - **Rand**
     - **Complete**
     - **F-M**
     - **Mutual**
     - **CPCC**
     - **IOA**
   * - Gower
     - 0.6333
     - 0.0399
     - 0.7948
     - 0.0007
     - **0.7162**
     - **0.8187**
   * - Euclidean
     - 0.6341
     - 0.0184
     - 0.7947
     - 0.0005
     - 0.7719
     - 0.8599
   * - Cosine
     - 0.6353
     - 0.1966
     - 0.7930
     - 0.0103
     - 0.9103
     - 0.9509
   * - Minkowski
     - 0.6237
     - **0.0045**
     - 0.7858
     - **0.0002**
     - 0.7851
     - 0.8702
   * - Dice
     - **0.5900**
     - 0.0195
     - **0.7421**
     - 0.0043
     - 0.8598
     - 0.9211
   * - Jaccard
     - 0.6402
     - 0.0295
     - 0.7998
     - **0.0002**
     - 0.8700
     - 0.9276

Car Insurance Claim Dataset

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - **Metric**
     - **Rand**
     - **Complete**
     - **F-M**
     - **Mutual**
     - **CPCC**
     - **IOA**
   * - Gower
     - 0.6096
     - 0.0350
     - 0.7691
     - 0.0041
     - **0.5539**
     - **0.6777**
   * - Euclidean
     - 0.4712
     - **0.0003**
     - 0.4740
     - **0.0003**
     - 0.6112
     - 0.7316
   * - Cosine
     - 0.6140
     - 0.1100
     - 0.7833
     - 0.0006
     - 0.6354
     - 0.7523
   * - Minkowski
     - **0.4680**
     - **0.0003**
     - **0.4639**
     - **0.0003**
     - 0.6306
     - 0.7476
   * - Dice
     - 0.5962
     - 0.0299
     - 0.7149
     - 0.0117
     - 0.6329
     - 0.7523
   * - Jaccard
     - 0.6098
     - 0.0193
     - 0.7796
     - 0.0004
     - 0.6394
     - 0.7564

Diabetes Dataset

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - **Metric**
     - **Rand**
     - **Complete**
     - **F-M**
     - **Mutual**
     - **CPCC**
     - **IOA**
   * - Gower
     - 0.5371
     - 0.0077
     - 0.6928
     - 0.0021
     - **0.6359**
     - **0.7490**
   * - Euclidean
     - 0.5530
     - 0.0318
     - 0.6903
     - 0.0103
     - 0.8456
     - 0.9105
   * - Cosine
     - **0.4994**
     - **0.0007**
     - **0.5186**
     - **0.0005**
     - 0.7340
     - 0.8320
   * - Minkowski
     - 0.5530
     - 0.0318
     - 0.6903
     - 0.0103
     - 0.8456
     - 0.9105
   * - Dice
     - 0.5455
     - 1.0000
     - 0.7386
     - NaN
     - NaN
     - NaN
   * - Jaccard
     - 0.5455
     - 1.0000
     - 0.7386
     - NaN
     - NaN
     - NaN


--------
HDBSCAN
--------


Adult Dataset

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - **Metric**
     - **Rand**
     - **Complete**
     - **F-M**
     - **Mutual**
   * - Gower
     - 0.3983
     - 0.0277
     - 0.2850
     - 0.0963
   * - Euclidean
     - 0.3701
     - **0.0127**
     - **0.1748**
     - 0.0596
   * - Cosine
     - 0.5075
     - 0.0316
     - 0.5421
     - **0.0436**
   * - Minkowski
     - **0.3674**
     - 0.0131
     - 0.1781
     - 0.0618
   * - Dice
     - 0.4260
     - 0.0507
     - 0.3718
     - 0.1081
   * - Jaccard
     - 0.4186
     - 0.0476
     - 0.3725
     - 0.0890

Car Insurance Claim Dataset

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - **Metric**
     - **Rand**
     - **Complete**
     - **F-M**
     - **Mutual**
   * - Gower
     - 0.5837
     - 0.0104
     - 0.6968
     - 0.0064
   * - Euclidean
     - 0.4653
     - 0.0085
     - 0.7806
     - 0.0218
   * - Cosine
     - 0.6077
     - 0.0095
     - 0.7621
     - **0.0016**
   * - Minkowski
     - 0.4729
     - **0.0076**
     - 0.4812
     - 0.0193
   * - Dice
     - 0.3972
     - 0.0198
     - **0.1298**
     - 0.0881
   * - Jaccard
     - **0.3945**
     - 0.0211
     - 0.1310
     - **0.0931**

Diabetes Dataset

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - **Metric**
     - **Rand**
     - **Complete**
     - **F-M**
     - **Mutual**
   * - Gower
     - 0.4996
     - **0.0006**
     - 0.5205
     - **0.0004**
   * - Euclidean
     - 0.5472
     - 0.0193
     - 0.6705
     - 0.0091
   * - Cosine
     - **0.4848**
     - 0.0064
     - **0.4190**
     - 0.0087
   * - Minkowski
     - 0.5480
     - 0.0189
     - 0.6718
     - 0.0089
   * - Dice
     - 0.5003
     - 0.0035
     - 0.5466
     - 0.0022
   * - Jaccard
     - 0.5068
     - 0.0013
     - 0.5525
     - 0.0009


---------------------------------
Suggested results interpretation
---------------------------------

Each of the analyzed datasets contains different types of data. For example, the adult.csv dataset includes both ratio and categorical data. On the other hand, diabetes.csv consists solely of numeric data, while car_insurance.csv incorporates an additional type: binary variables. Additionally, all data files have been labeled accordingly to facilitate analysis.
As demonstrated by the results, the custom Gower metric implementation achieves the highest performance in hierarchical clustering, as optimized by CPCC and IoA. This indicates that optimizing the weights yields superior scores, despite significant effort being invested in balancing variable importance without weight adjustments. However, it is important to note that weight optimization must be repeated following any changes to the dataset or when selecting a different subset of features.
However, this does not imply that the Gower metric is inherently the best for all scenarios. One should consider whether minimizing the metric value always translates to the most meaningful clustering or classification results. Over-reliance on achieving minimal distance values might lead to an over-clustering effect, identifying patterns where none should exist.
Thus, the best approach is to select a distance metric that best aligns with the specific problem at hand.