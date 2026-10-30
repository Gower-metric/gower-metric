# Copyright (c) 2025 - 2026 the gower-metric developers
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from gower_metric._typing import ObjectArray, Record


def to_array(
    record: Record,
) -> ObjectArray:
    """Convert a record to a flat NumPy array of dtype object.

    Args:
        record (Record): feature values.

    Returns:
        ObjectArray: 1D array of feature values, with original dtype preserved.

    """
    if isinstance(record, np.ndarray):
        return np.asarray(record.flatten(), dtype=object)
    if isinstance(record, pd.Series):
        return np.asarray(record.to_numpy(dtype=object))
    return np.asarray(record, dtype=object)
