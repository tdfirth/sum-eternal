"""
Chapter 1: First Blood
======================
Basic contractions — your first steps into the notation.

The title screen fades. A single vector appears.
Your journey into the notation begins.

Complete these functions using jax.numpy.einsum to proceed.
Run tests with: uv run pytest tests/test_c01_first_blood.py -v
"""

import jax.numpy as jnp


def vector_sum(v):
    """
    Sum all elements of a vector.

    Args:
        v: array of shape (n,)

    Returns:
        Scalar (shape ()) — the sum of all elements

    Example:
        >>> vector_sum(jnp.array([1.0, 2.0, 3.0]))
        6.0

    Hint:
        When an index appears on the input but not the output, what happens?
        You have one dimension going in, zero dimensions coming out.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Time to rip and tensor.")


def element_multiply(a, b):
    """
    Element-wise multiplication of two vectors.

    Args:
        a: array of shape (n,)
        b: array of shape (n,)

    Returns:
        Array of shape (n,) — element-wise product a[i] * b[i]

    Example:
        >>> element_multiply(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        [3.0, 8.0]

    Hint:
        Both vectors use the same positions. Nothing disappears.
        What index letter would you use? Does it survive to the output?
    """
    # YOUR CODE HERE
    raise NotImplementedError("The notation awaits.")


def dot_product(a, b):
    """
    Compute the dot product of two vectors.

    Args:
        a: array of shape (n,)
        b: array of shape (n,)

    Returns:
        Scalar (shape ()) — the dot product sum(a[i] * b[i])

    Example:
        >>> dot_product(jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))
        32.0  # 1*4 + 2*5 + 3*6

    Hint:
        You multiply matching positions, then add them all up.
        The index appears twice on the left. What's on the right?
    """
    # YOUR CODE HERE
    raise NotImplementedError("Indices aligned. Execute.")


def outer_product(a, b):
    """
    Compute the outer product of two vectors.

    Args:
        a: array of shape (n,)
        b: array of shape (m,)

    Returns:
        Array of shape (n, m) — where result[i,j] = a[i] * b[j]

    Example:
        >>> outer_product(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0, 5.0]))
        [[3.0, 4.0, 5.0],
         [6.0, 8.0, 10.0]]

    Hint:
        Each input has its own dimension. The output has both.
        If 'a' has index i and 'b' has index j, what shape is the result?
    """
    # YOUR CODE HERE
    raise NotImplementedError("Expand your dimensions.")


def matrix_vector_mul(M, v):
    """
    Multiply a matrix by a vector.

    Args:
        M: array of shape (n, m)
        v: array of shape (m,)

    Returns:
        Array of shape (n,) — the matrix-vector product

    Example:
        >>> M = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> v = jnp.array([1.0, 1.0])
        >>> matrix_vector_mul(M, v)
        [3.0, 7.0]

    Hint:
        M has rows and columns. v matches M's columns.
        Which dimension gets contracted away? Which survives?
    """
    # YOUR CODE HERE
    raise NotImplementedError("Contract the inner dimension.")


def matrix_matrix_mul(A, B):
    """
    Multiply two matrices.

    Args:
        A: array of shape (n, m)
        B: array of shape (m, p)

    Returns:
        Array of shape (n, p) — the matrix product

    Example:
        >>> A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        >>> matrix_matrix_mul(A, B)
        [[19.0, 22.0],
         [43.0, 50.0]]

    Hint:
        A is (n,m), B is (m,p). The 'm' dimension is shared.
        What happens to shared indices? What shape is the output?
    """
    # YOUR CODE HERE
    raise NotImplementedError("FIRST BLOOD. Complete the contraction.")
