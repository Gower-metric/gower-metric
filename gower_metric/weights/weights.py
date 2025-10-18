import numpy as np


def get_weights(
    n_features: int, config: dict[int, float] | str | None = None
) -> np.ndarray:
    """
    Get weights for features based on the provided configuration.

    Args:
        n_features (int): Number of features.
        config (dict[int, float] | str | None): Configuration for weights. If None or "uniform", all features will have equal weight of 1.
            If a dictionary, keys are feature indices and values are weights.

    Returns:
        np.ndarray: Array of weights for each feature.

    Raises:
        ValueError: If config is not None, "uniform", or a dictionary.
    """
    w = np.ones(n_features, dtype=float)

    if isinstance(config, dict):
        for idx, val in config.items():
            w[idx] = float(val)

    elif config is None or config == "uniform":
        pass

    else:
        raise ValueError(
            "config must be None, 'uniform', or a dictionary mapping feature "
            f"indices to weights, got {type(config).__name__}"
        )

    return w
