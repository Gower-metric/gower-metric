import numpy as np
import pandas as pd


def all_ones_off_diagonal(X: pd.DataFrame | np.ndarray) -> bool:
    """
    Return True if all off-diagonal elements are 1 (diagonal ignored).

    Args:
        X (np.ndarray | pd.DataFrame): shape of (n_samples, n_features).
            For DataFrame inputs, column names in feature_types are converted to indices.

    Returns:
        bool: True if Gower distance between all pairs of samples is equal to 1.

    Raises:
        TypeError: If X not a DataFrame or ndarray.

    Example:
        >>> from sklearn.metrics import pairwise_distances
        >>> from gower_metric import Gower
        >>> from gower_metric.core.config import Config
        >>> from gower_metric.utils.aux import all_ones_off_diagonal
        >>> data = pd.DataFrame({
        ...     'feature1': [[1.0], [2.0], [3.0], [4.0]],
        ...     'feature2': ['A', 'B', 'A', 'C'],
        ... })
        >>> cfg = Config(
        ...     feature_types={0: 'numeric_interval', 1: 'categorical_nominal'},
        ... )
        >>> gower = Gower(cfg).fit(data)
        >>> transformed_data = gower.transform(data)
        >>> pairwise_dist_result = pairwise_distances(transformed_data, metric=gower)
        >>> all_ones_off_diagonal(pairwise_dist_result)
    """
    if isinstance(X, pd.DataFrame):
        arr = X.values
    elif isinstance(X, np.ndarray):
        arr = X
    else:
        raise TypeError(f"Expected DataFrame or ndarray, got {type(X).__name__}")

    mask = ~np.eye(arr.shape[0], dtype=bool)
    return bool((arr[mask] == 1).all())
