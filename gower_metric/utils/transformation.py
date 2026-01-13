import numpy as np
import pandas as pd

from gower_metric.core.exceptions import IllegalStateError, IllegalTypeError


def validate_if_transformed(X: np.ndarray | pd.DataFrame) -> None:
    """Validate if the input data is transformed.

    Args:
        X (np.ndarray | pd.DataFrame): Input data.

    Raises:
        IllegalTypeError: If the input data is not a numpy array or pandas DataFrame.
        IllegalStateError: If the input data is not transformed.

    Reference:
        - `Pandas dataframe attributes <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.attrs.html>`_
        - `Numpy dtype metadata <https://numpy.org/doc/stable/reference/generated/numpy.dtype.metadata.html>`_

    """
    if isinstance(X, pd.DataFrame):
        try:
            if not X.attrs.get("transformed", True):
                msg = "Input data must be a transformed pandas DataFrame."
                raise IllegalStateError(msg)
        except AttributeError:
            msg = "Input data must be a transformed pandas DataFrame."
            raise IllegalStateError(msg) from None
    elif isinstance(X, np.ndarray):
        try:
            if not X.dtype.metadata or not X.dtype.metadata.get("transformed", True):
                msg = "Input data must be a transformed numpy array."
                raise IllegalStateError(msg)
        except AttributeError:
            msg = "Input data must be a transformed numpy array."
            raise IllegalStateError(msg) from None
    else:
        msg = "Input data must be a numpy array or pandas DataFrame."
        raise IllegalTypeError(msg)
