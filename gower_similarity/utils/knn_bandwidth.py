import numpy as np

from typing import Optional

def knn_bandwidth(x: np.ndarray, k: Optional[int] = None) -> float:
    """
    Compute the k-nearest neighbor bandwidth for a 1D array.

    Args:
        x: 1D array of data points
        k: number of nearest neighbors to consider. If k is None or less than 1, 
            it will be set to the square root of the number of points.

    Returns:
        Bandwidth value (float)
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size

    if n <= 1:
        return 0.0
    
    k = int(np.sqrt(n)) if (k is None or k < 1) else k
    k = min(k, n - 1)
    diffs = np.abs(x[k:] - x[:-k])

    return np.median(diffs)