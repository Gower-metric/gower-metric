import warnings

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix


def __get_csr_matrix(
    data: np.ndarray,
    data_type: type[np.floating],
) -> csr_matrix:
    """Convert a dense array to a CSR sparse matrix.

    Args:
        data (np.ndarray): dense array (n_samples, n_samples).
        data_type (type[np.floating]): data type for the output sparse matrix.

    Returns:
        csr_matrix: CSR sparse matrix.

    """
    return csr_matrix(data, dtype=data_type)


def __get_csc_matrix(
    data: np.ndarray,
    data_type: type[np.floating],
) -> csc_matrix:
    """Convert a dense array to a CSC sparse matrix.

    Args:
        data (np.ndarray): dense array (n_samples, n_samples).
        data_type (type[np.floating]): data type for the output sparse matrix.

    Returns:
        csc_matrix: CSC sparse matrix.

    """
    return csc_matrix(data, dtype=data_type)


def __get_coo_matrix(
    data: np.ndarray,
    data_type: type[np.floating],
) -> coo_matrix:
    """Convert a dense array to a COO sparse matrix.

    Args:
         data (np.ndarray): dense array (n_samples, n_samples).
         data_type (type[np.floating]): data type for the output sparse matrix.

    Returns:
        coo_matrix: COO sparse matrix.

    """
    return coo_matrix(data, dtype=data_type)


def get_scipy_sparse_matrix(
    data: np.ndarray,
    matrix_format: str = "csr",
    data_type: type[np.floating] = np.float32,
) -> csr_matrix | csc_matrix | coo_matrix:
    """Convert a dense array to a specified format of sparse matrix.

    Args:
        data (np.ndarray): dense array (n_samples, n_samples).
        matrix_format (str): Format of the output sparse matrix. Options are 'csr', 'csc', 'coo'. Default is 'csr'.
        data_type (type[np.floating]): data type for the output sparse matrix. Default is np.float32.

    Returns:
        csr_matrix | csc_matrix | coo_matrix: Sparse matrix in the specified format.

    Raises:
        ValueError: If an unsupported matrix format is provided.

    """
    sparse_dtype = data_type
    if issubclass(data_type, np.float16):
        sparse_dtype = np.float32
        warnings.warn(
            f"scipy.sparse does not support {data_type.__name__}. "
            f"Sparse matrix will use {sparse_dtype.__name__} instead.",
            UserWarning,
            stacklevel=2,
        )

    if matrix_format == "csr":
        return __get_csr_matrix(data, sparse_dtype)
    if matrix_format == "csc":
        return __get_csc_matrix(data, sparse_dtype)
    if matrix_format == "coo":
        return __get_coo_matrix(data, sparse_dtype)
    msg = (
        f"Unsupported matrix format: {matrix_format}. "
        "Supported formats are 'csr', 'csc', 'coo'."
    )
    raise ValueError(
        msg,
    )
