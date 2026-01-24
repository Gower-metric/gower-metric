import numpy as np
import pandas as pd

from gower_metric.core.exceptions import IllegalStateError


def validate_if_transformed(X: pd.DataFrame | pd.Series | np.ndarray) -> None:
    """Validate if the input data is transformed.

    Numpy arrays are not validated due to not officially supported metadata handling.

    Args:
        X (pd.DataFrame | pd.Series): Input data.

    Raises:
        IllegalStateError: If the input data is not transformed.

    Reference:
        - `Pandas dataframe attributes <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.attrs.html>`_

    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        try:
            if not X.attrs.get("transformed", False):
                msg = "Input data must be transformed."
                raise IllegalStateError(msg)
        except AttributeError:
            msg = "Input data must be transformed."
            raise IllegalStateError(msg) from None
