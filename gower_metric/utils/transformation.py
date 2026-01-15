import pandas as pd

from gower_metric.core.exceptions import IllegalStateError


def validate_if_transformed(X: pd.DataFrame | pd.Series) -> None:
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
    else:
        pass


def validate_if_double_transformed(is_transformed: bool) -> None:
    """Validate if the input data has already been transformed.

    Args:
        is_transformed (bool): Whether the input data has already been transformed.

    Raises:
        IllegalStateError: If the input data has already been transformed.

    """
    if is_transformed:
        msg = "Input data has already been transformed!"
        raise IllegalStateError(msg)
