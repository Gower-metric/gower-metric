=============
Scikit-learn
=============

As previously mentioned, user needs to call ``transform`` method to ensure that the data
contains only numerical values before passing it to scikit-learn functions.

-------------------
Pairwise distances
-------------------

.. code-block:: python

    from sklearn.metrics import pairwise_distances
    from gower_metric import Gower

    gower = Gower(feature_types=feature_types).fit(df)
    df_num = gower.transform(df)

    matrix_scikit = pairwise_distances(df_num, metric=gower, n_jobs=-1, ensure_all_finite=False)

.. note::
    When using ``pairwise_distances`` from scikit-learn, make sure to set
    ``ensure_all_finite=False`` to avoid errors due to custom handling of NaN values.

---------------------
KNeighborsClassifier
---------------------

.. code-block:: python

    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from gower_metric import Gower

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    gower = Gower(feature_types=feature_types).fit(X_train)
    X_train = gower.transform(X_train)
    X_test = gower.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5, metric=gower, n_jobs=-1)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    # you can now evaluate and print the results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

---------------------
Scikit-learn HDBSCAN
---------------------

.. code-block:: python

    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import silhouette_score
    from gower_metric import Gower

    gower = Gower(feature_types=feature_types).fit(df)
    df_transformed = gower.transform(df)

    clusterer = HDBSCAN(metric=gower, min_cluster_size=10, min_samples=5)
    clusterer.fit(df_transformed)

    labels = clusterer.labels_
    silhouette_avg = silhouette_score(df_transformed, labels)
    print(f"Silhouette Score: {silhouette_avg:.2f}")