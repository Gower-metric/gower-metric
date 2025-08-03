import numpy as np


def silverman_bandwidth(x: np.ndarray) -> float:
    """
    Calculate Silverman's rule of thumb bandwidth for kernel density estimation.

    Args:
        x: Input data array.

    Returns:
        float: Calculated bandwidth, ensuring it is non-negative.
    """
    x = x[~np.isnan(x)]
    n = x.size

    if n < 2:
        return 0.0

    s = x.std(ddof=1, axis=0)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    h = 1.06 * min(s, iqr / 1.34) * n ** (-1 / 5)

    return max(h, 0.0)
