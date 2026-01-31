"""
Chapter 2: Knee Deep in the Indices
===================================
Reductions & rearrangements — matrices bend to your will.

The debug view expands. You're manipulating matrices now.
The indices dance at your command.

Complete these functions using jax.numpy.einsum to proceed.
Run tests with: uv run pytest tests/test_c02_knee_deep_in_the_indices.py -v
"""

import jax.numpy as jnp


def transpose(M):
    """
    Transpose a matrix.

    Args:
        M: array of shape (n, m)

    Returns:
        Array of shape (m, n) — the transposed matrix

    Example:
        >>> M = jnp.array([[1, 2, 3], [4, 5, 6]])
        >>> transpose(M)
        [[1, 4],
         [2, 5],
         [3, 6]]

    Hint:
        Just swap the order of indices on the output.
        Pattern: 'ij->ji'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Flip the dimensions.")


def trace(M):
    """
    Compute the trace of a square matrix (sum of diagonal elements).

    Args:
        M: array of shape (n, n) — must be square

    Returns:
        Scalar — the sum of diagonal elements M[i,i]

    Example:
        >>> M = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> trace(M)
        15.0  # 1 + 5 + 9

    Hint:
        Use the same index twice to select the diagonal.
        Then sum by not including it in the output.
        Pattern: 'ii->'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Follow the diagonal.")


def diag_extract(M):
    """
    Extract the diagonal of a square matrix.

    Args:
        M: array of shape (n, n) — must be square

    Returns:
        Array of shape (n,) — the diagonal elements

    Example:
        >>> M = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> diag_extract(M)
        [1, 5, 9]

    Hint:
        Same index twice selects the diagonal.
        Keep the index in the output to preserve the elements.
        Pattern: 'ii->i'
    """
    # YOUR CODE HERE
    raise NotImplementedError("The diagonal reveals itself.")


def sum_rows(M):
    """
    Sum each row of a matrix.

    Args:
        M: array of shape (n, m)

    Returns:
        Array of shape (n,) — sum of each row

    Example:
        >>> M = jnp.array([[1, 2, 3], [4, 5, 6]])
        >>> sum_rows(M)
        [6, 15]

    Hint:
        Keep the row index, sum away the column index.
        Pattern: 'ij->i'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Collapse the columns.")


def sum_cols(M):
    """
    Sum each column of a matrix.

    Args:
        M: array of shape (n, m)

    Returns:
        Array of shape (m,) — sum of each column

    Example:
        >>> M = jnp.array([[1, 2, 3], [4, 5, 6]])
        >>> sum_cols(M)
        [5, 7, 9]

    Hint:
        Keep the column index, sum away the row index.
        Pattern: 'ij->j'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Collapse the rows.")


def frobenius_norm_sq(M):
    """
    Compute the squared Frobenius norm of a matrix.

    The Frobenius norm is the square root of the sum of squared elements.
    This function returns the squared norm (sum of squared elements).

    Args:
        M: array of shape (n, m)

    Returns:
        Scalar — sum of M[i,j]^2 for all i,j

    Example:
        >>> M = jnp.array([[1, 2], [3, 4]])
        >>> frobenius_norm_sq(M)
        30.0  # 1 + 4 + 9 + 16

    Hint:
        Element-wise multiply M with itself, then sum everything.
        Pattern: 'ij,ij->'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Square and sum. The norm approaches.")
