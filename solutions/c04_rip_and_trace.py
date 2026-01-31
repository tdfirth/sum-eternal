"""
Chapter 4: Rip and Trace
========================
Ray generation — the raycaster awakens.

Before you can see, you must cast forth your vision.
Each ray a question. Each wall an answer.

Complete these functions using jax.numpy.einsum to proceed.
Run tests with: uv run pytest tests/test_c04_rip_and_trace.py -v
"""

import jax.numpy as jnp


def angles_to_directions(angles):
    """
    Convert angles to unit direction vectors.

    Args:
        angles: array of shape (r,) — angles in radians
                0 = east (+x), π/2 = north (+y)

    Returns:
        Array of shape (r, 2) — unit direction vectors [cos(θ), sin(θ)]

    Example:
        >>> angles = jnp.array([0, jnp.pi/2, jnp.pi])
        >>> angles_to_directions(angles)
        [[1, 0], [0, 1], [-1, 0]]  # approximately

    Hint:
        Stack cos and sin along a new axis.
        This isn't a pure einsum problem — use jnp.stack or jnp.column_stack.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Cast your rays into the void.")


def rotate_vectors(vecs, angle):
    """
    Rotate 2D vectors by a given angle.

    Args:
        vecs: array of shape (n, 2) — vectors to rotate
        angle: scalar — rotation angle in radians (counter-clockwise)

    Returns:
        Array of shape (n, 2) — rotated vectors

    Example:
        >>> vecs = jnp.array([[1, 0], [0, 1]])
        >>> rotate_vectors(vecs, jnp.pi/2)  # 90 degrees
        [[0, 1], [-1, 0]]  # approximately

    Hint:
        Build a 2x2 rotation matrix:
        [[cos(θ), -sin(θ)],
         [sin(θ),  cos(θ)]]

        Then apply it to each vector. The einsum pattern is 'ij,nj->ni'.
        (Matrix indices ij, n vectors with j components, output n vectors with i components)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Rotate your perspective.")


def normalize_vectors(v):
    """
    Normalize a batch of vectors to unit length.

    Args:
        v: array of shape (n, 2) — vectors to normalize

    Returns:
        Array of shape (n, 2) — unit vectors (same direction, length 1)

    Example:
        >>> v = jnp.array([[3, 4], [0, 5]])
        >>> normalize_vectors(v)
        [[0.6, 0.8], [0, 1]]

    Hint:
        Compute magnitude using 'ni,ni->n' (squared, then sqrt).
        Divide each component by its vector's magnitude.
        Watch out for zero-length vectors (add small epsilon).
    """
    # YOUR CODE HERE
    raise NotImplementedError("Unit vectors. Perfect direction.")


def scale_vectors(v, scales):
    """
    Scale each vector by its corresponding scalar.

    Args:
        v: array of shape (n, d) — n vectors of dimension d
        scales: array of shape (n,) — scale factor for each vector

    Returns:
        Array of shape (n, d) — scaled vectors

    Example:
        >>> v = jnp.array([[1, 2], [3, 4]])
        >>> scales = jnp.array([2, 0.5])
        >>> scale_vectors(v, scales)
        [[2, 4], [1.5, 2]]

    Hint:
        Each scale (a scalar per vector) multiplies all components of that vector.
        The scale broadcasts across the dimension 'd'. What indices match?
    """
    # YOUR CODE HERE
    raise NotImplementedError("Scale the rays. Control the reach.")
