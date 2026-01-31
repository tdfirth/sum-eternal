"""
Chapter 3: The Slaughter Batch
==============================
Batch operations — one operation, many data.

This is the key insight. The Einsteins operated in batches.
So must you. Process everything at once.

Complete these functions using jax.numpy.einsum to proceed.
Run tests with: uv run pytest tests/test_c03_the_slaughter_batch.py -v
"""

import jax.numpy as jnp


def batch_vector_sum(batch):
    """
    Sum each vector in a batch.

    Args:
        batch: array of shape (b, n) — b vectors of length n

    Returns:
        Array of shape (b,) — sum of each vector

    Example:
        >>> batch = jnp.array([[1, 2, 3], [4, 5, 6]])
        >>> batch_vector_sum(batch)
        [6, 15]

    Hint:
        The batch dimension survives, the vector dimension gets summed.
        Pattern: 'bi->b'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Sum them all. At once.")


def batch_dot_pairwise(a, b):
    """
    Compute pairwise dot products between corresponding vectors in two batches.

    Args:
        a: array of shape (b, n) — batch of b vectors
        b: array of shape (b, n) — batch of b vectors (same shapes)

    Returns:
        Array of shape (b,) — dot product of a[i] with b[i] for each i

    Example:
        >>> a = jnp.array([[1, 2], [3, 4]])
        >>> b = jnp.array([[5, 6], [7, 8]])
        >>> batch_dot_pairwise(a, b)
        [17, 53]  # [1*5+2*6, 3*7+4*8]

    Hint:
        Same batch index, same vector index, both get matched.
        Sum over the vector dimension, keep the batch dimension.
        Pattern: 'bi,bi->b'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Pairwise. Parallel. Perfect.")


def batch_magnitude_sq(v):
    """
    Compute squared magnitude of each vector in a batch.

    Args:
        v: array of shape (b, n) — batch of b vectors

    Returns:
        Array of shape (b,) — squared magnitude of each vector

    Example:
        >>> v = jnp.array([[3, 4], [5, 12]])
        >>> batch_magnitude_sq(v)
        [25, 169]  # [3^2+4^2, 5^2+12^2]

    Hint:
        Dot product of each vector with itself.
        Pattern: 'bi,bi->b'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Square the components. Sum the results.")


def all_pairs_dot(a, b):
    """
    Compute dot products between ALL pairs of vectors from two batches.

    Args:
        a: array of shape (n, d) — n vectors of dimension d
        b: array of shape (m, d) — m vectors of dimension d

    Returns:
        Array of shape (n, m) — result[i,j] = dot(a[i], b[j])

    Example:
        >>> a = jnp.array([[1, 0], [0, 1]])
        >>> b = jnp.array([[1, 1], [2, 3]])
        >>> all_pairs_dot(a, b)
        [[1, 2],   # [1*1+0*1, 1*2+0*3]
         [1, 3]]   # [0*1+1*1, 0*2+1*3]

    Hint:
        Different batch indices (i for a, j for b), same dimension index (d).
        Sum over d, keep i and j.
        Pattern: 'id,jd->ij'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Every pair. No exceptions.")


def batch_matrix_vector(M, batch):
    """
    Apply a matrix to each vector in a batch.

    Args:
        M: array of shape (d, d) — a square matrix
        batch: array of shape (b, d) — b vectors of dimension d

    Returns:
        Array of shape (b, d) — M applied to each vector

    Example:
        >>> M = jnp.array([[0, -1], [1, 0]])  # 90-degree rotation
        >>> batch = jnp.array([[1, 0], [0, 1], [1, 1]])
        >>> batch_matrix_vector(M, batch)
        [[0, 1], [-1, 0], [-1, 1]]

    Hint:
        Matrix indices are ij, batch vector indices are bj.
        Contract over j, keep b and i (which becomes the new component index).
        Pattern: 'ij,bj->bi'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Transform them all. Simultaneously.")


def batch_outer(a, b):
    """
    Compute outer product for each pair of corresponding vectors in two batches.

    Args:
        a: array of shape (b, n) — batch of b vectors
        b: array of shape (b, m) — batch of b vectors

    Returns:
        Array of shape (b, n, m) — outer product for each batch element

    Example:
        >>> a = jnp.array([[1, 2], [3, 4]])
        >>> b = jnp.array([[5, 6, 7], [8, 9, 10]])
        >>> batch_outer(a, b)
        [[[5, 6, 7], [10, 12, 14]],
         [[24, 27, 30], [32, 36, 40]]]

    Hint:
        Same batch index, different vector indices.
        Pattern: 'bi,bj->bij'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Outer products. In parallel. The batch is complete.")
