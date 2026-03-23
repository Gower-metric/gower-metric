import numpy as np


def apply_missing_strategy(
    diff: np.ndarray,
    present: np.ndarray,
    missing_strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the chosen missing-values strategy to the raw diff matrix.

    Args:
        diff (np.ndarray): raw distance matrix for one feature, shape (n_x, n_y).
        present (np.ndarray): boolean mask where True means both values were non-missing.
        missing_strategy (str): one of "ignore", "max_dist", "raise_error".

    Returns:
        tuple[np.ndarray, np.ndarray]: Diff is adjusted distance matrix. Count_mask is int matrix of same shape, how much to add to count_present.

    Raises:
        ValueError: if missing_strategy is not recognized.

    """
    if missing_strategy == "ignore":
        diff[~present] = 0.0
        count_mask = present.astype(int)

    elif missing_strategy == "max_dist":
        diff[~present] = 1.0
        count_mask = np.ones_like(present, dtype=int)

    elif missing_strategy == "raise_error":
        if not present.all():
            msg = (
                "Missing values detected in data. "
                "Set missing_strategy='ignore' or 'max_dist' to handle them."
            )
            raise ValueError(msg)
        count_mask = present.astype(int)

    else:
        msg = f"Unknown missing_strategy '{missing_strategy}'"
        raise ValueError(msg)

    return diff, count_mask
