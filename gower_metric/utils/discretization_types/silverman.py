import numpy as np

MIN_SAMPLES = 2


def silverman_bandwidth(x: np.ndarray, c: float = 1.06) -> float:
    """Calculate Silverman's rule of thumb bandwidth for kernel density estimation.

    Args:
        x (np.ndarray): Input data array.
        c (int | float): Silverman constant in the formula
            ``h = c * min(s, IQR / 1.34) * n ** (-1/5)``. Default ``1.06``.

    Returns:
        float: Calculated bandwidth, ensuring it is non-negative.

    """
    x = x[~np.isnan(x)]
    n = x.size

    if n < MIN_SAMPLES:
        return 0.0

    s = x.std(ddof=1, axis=0)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    h = c * min(s, iqr / 1.34) * n ** (-1 / 5)

    return float(max(h, 0.0))
