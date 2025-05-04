import numpy as np

from typing import Dict, Optional, Union

def get_weights(
        n_features: int, 
        config: Optional[Union[Dict[int, float], str]] = None
    ) -> np.ndarray:
    """
    Get weights for features based on the provided configuration.

    Args:
        n_features (int): Number of features.
        config (Optional[Union[Dict[int, float], str]]): Configuration for weights.
            - If None or "uniform", all features will have equal weight of 1.
            - If a dictionary, keys are feature indices and values are weights.

    Returns:
        np.ndarray: Array of weights for each feature.
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
            "indices to weights, got {type(config).__name__}"
        )

    return w