"""
Chapter 8: The Icon of Ein
==========================
Combat — your weapon is mathematics.

Your ammunition is understanding. When you sum an Einstein,
you do not destroy — you demonstrate. Fire the notation.
Reduce them to scalars.

Complete these functions using jax.numpy.einsum to proceed.
Run tests with: uv run pytest tests/test_c08_the_icon_of_ein.py -v
"""

import jax.numpy as jnp


def project_points_onto_ray(points, origin, direction):
    """
    Project points onto a ray, returning distance along the ray.

    For each point P, find t such that the projection of (P - origin)
    onto the ray direction gives the closest point on the ray.

    Args:
        points: array of shape (n, 2) — points to project
        origin: array of shape (2,) — ray origin
        direction: array of shape (2,) — ray direction (should be unit length)

    Returns:
        Array of shape (n,) — signed distance along ray to projection
            Positive = in front, negative = behind

    Example:
        >>> points = jnp.array([[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        >>> origin = jnp.array([0.0, 0.0])
        >>> direction = jnp.array([1.0, 0.0])  # pointing east
        >>> project_points_onto_ray(points, origin, direction)
        [2.0, 0.0, -1.0]

    Hint:
        t = (P - origin) · direction
        Use einsum: 'ni,i->n' for the dot product of each point with direction.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Project onto the ray. Find the distance.")


def perpendicular_distance_to_ray(points, origin, direction):
    """
    Compute perpendicular distance from each point to a ray.

    This tells us how far "off to the side" each point is from the ray.
    Used for hit detection — if perpendicular distance < radius, it's a hit.

    Args:
        points: array of shape (n, 2) — points to measure
        origin: array of shape (2,) — ray origin
        direction: array of shape (2,) — ray direction (should be unit length)

    Returns:
        Array of shape (n,) — perpendicular distance (always non-negative)

    Example:
        >>> points = jnp.array([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
        >>> origin = jnp.array([0.0, 0.0])
        >>> direction = jnp.array([1.0, 0.0])  # pointing east
        >>> perpendicular_distance_to_ray(points, origin, direction)
        [1.0, 0.0, 3.0]

    Hint:
        Use the 2D cross product magnitude:
        |perp_dist| = |(P - origin) × direction|

        For 2D: cross(a, b) = a[0]*b[1] - a[1]*b[0]
        Compute this for each point.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Perpendicular distance. How close is close enough?")


def ray_hits_target(proj_dist, perp_dist, radii, wall_dist):
    """
    Determine which targets are hit by a ray.

    A target is hit if:
    1. It's in front of the player (proj_dist > 0)
    2. It's closer than the nearest wall (proj_dist < wall_dist)
    3. The perpendicular distance is less than its radius

    Args:
        proj_dist: array of shape (n,) — distance along ray to each target
        perp_dist: array of shape (n,) — perpendicular distance to each target
        radii: array of shape (n,) — hit radius for each target
        wall_dist: scalar — distance to nearest wall

    Returns:
        Array of shape (n,) — boolean mask (True = hit)

    Example:
        >>> proj_dist = jnp.array([5.0, 15.0, 3.0])
        >>> perp_dist = jnp.array([0.5, 0.2, 2.0])
        >>> radii = jnp.array([1.0, 1.0, 1.0])
        >>> ray_hits_target(proj_dist, perp_dist, radii, 10.0)
        [True, False, False]
        # First: in front, before wall, close enough
        # Second: in front, but behind wall
        # Third: in front, before wall, but too far off-axis

    Hint:
        Combine three conditions with logical AND.
        No einsum needed — just boolean operations.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Hit detection. The math of combat.")


def move_toward_point(positions, target, speeds, dt):
    """
    Move positions toward a target point.

    Each position moves toward the target at its own speed.

    Args:
        positions: array of shape (n, 2) — current positions
        target: array of shape (2,) — target position
        speeds: array of shape (n,) — movement speed for each entity
        dt: scalar — time step

    Returns:
        Array of shape (n, 2) — new positions

    Example:
        >>> positions = jnp.array([[10.0, 0.0], [0.0, 10.0]])
        >>> target = jnp.array([0.0, 0.0])
        >>> speeds = jnp.array([1.0, 2.0])
        >>> move_toward_point(positions, target, speeds, 1.0)
        [[9.0, 0.0], [0.0, 8.0]]  # approximately

    Hint:
        1. Compute direction to target: target - positions (shape: n, 2)
        2. Normalize each direction vector
        3. Scale by speed * dt: use 'ni,n->ni' or broadcasting
        4. Add to positions
    """
    # YOUR CODE HERE
    raise NotImplementedError("The Einsteins approach. They always approach.")
