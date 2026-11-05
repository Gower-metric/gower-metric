# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

from typing import Final

import numpy as np

from gower_metric._typing import FloatArray

NAME: Final = "knn"


def bandwidth(x: FloatArray, k: int | None = None) -> float:
    """Compute the k-nearest neighbor bandwidth for a 1D array.

    Args:
        x (FloatArray): 1D array of data points
        k (Optional[int]): number of nearest neighbors to consider. If k is None or less than 1, it will be set to the square root of the number of points.

    Returns:
        float:
            - bandwidth value

    """
    x = np.sort(np.asarray(x, dtype=float))
    x = x[~np.isnan(x)]
    n = x.size

    if n <= 1:
        return 0.0

    k = int(np.sqrt(n)) if (k is None or k < 1) else k
    k = min(k, n - 1)
    diffs = x[k:] - x[:-k]

    return float(np.median(diffs))
