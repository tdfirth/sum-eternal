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
        Rows become columns, columns become rows.
        How do you reorder the indices on the output side?
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
        The diagonal is where row index equals column index.
        How do you express "same position" in einsum? What if nothing survives?
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
        Like trace, but you want to keep the values, not sum them.
        What changes on the output side?
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
        You want one value per row. Which index represents rows?
        Which index should disappear (get summed)?
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
        You want one value per column. Which index represents columns?
        The opposite of sum_rows.
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
        You need M squared element-wise, then summed.
        How do you multiply a matrix with itself in einsum? What survives?
    """
    # YOUR CODE HERE
    raise NotImplementedError("Square and sum. The norm approaches.")
