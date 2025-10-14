import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix


def __get_csr_matrix(
    data: np.ndarray, data_type: type[np.floating | np.integer]
) -> csr_matrix:
    """Convert a dense array to a CSR sparse matrix.

    Args:
        data: Dense array (n_samples, n_samples).
        data_type: Data type for the output sparse matrix.

    Returns:
        CSR sparse matrix.
    """
    return csr_matrix(data, dtype=data_type)


def __get_csc_matrix(
    data: np.ndarray, data_type: type[np.floating | np.integer]
) -> csc_matrix:
    """Convert a dense array to a CSC sparse matrix.

    Args:
        data: Dense array (n_samples, n_samples).
        data_type: Data type for the output sparse matrix.

    Returns:
        CSC sparse matrix.
    """
    return csc_matrix(data, dtype=data_type)


def __get_coo_matrix(
    data: np.ndarray, data_type: type[np.floating | np.integer]
) -> coo_matrix:
    """Convert a dense array to a COO sparse matrix.

    Args:
         data: Dense array (n_samples, n_samples).
         data_type: Data type for the output sparse matrix.

    Returns:
         COO sparse matrix.
    """
    return coo_matrix(data, dtype=data_type)


def get_scipy_sparse_matrix(
    data: np.ndarray,
    matrix_format: str = "csr",
    data_type: type[np.floating | np.integer] = np.float32,
) -> csr_matrix | csc_matrix | coo_matrix:
    """Convert a dense array to a specified format of sparse matrix.

    Args:
        data: Dense array (n_samples, n_samples).
        matrix_format: Format of the output sparse matrix. Options are 'csr', 'csc', 'coo'.
        data_type: Data type for the output sparse matrix.

    Returns:
        Sparse matrix in the specified format.

    Raises:
        ValueError: If an unsupported matrix format is provided.
    """
    if matrix_format == "csr":
        return __get_csr_matrix(data, data_type)
    elif matrix_format == "csc":
        return __get_csc_matrix(data, data_type)
    elif matrix_format == "coo":
        return __get_coo_matrix(data, data_type)
    else:
        raise ValueError(
            f"Unsupported matrix format: {matrix_format}. "
            "Supported formats are 'csr', 'csc', 'coo'."
        )
