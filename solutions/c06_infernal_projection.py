"""
Chapter 6: Infernal Projection
==============================
Screen rendering — distances become pixels.

The abstract becomes visible. Your notation made manifest.
Transform raw distances into the 3D view you've been building toward.

Complete these functions using jax.numpy.einsum to proceed.
Run tests with: uv run pytest tests/test_c06_infernal_projection.py -v
"""

import jax.numpy as jnp


def fisheye_correct(dists, angles, player_angle):
    """
    Correct distances for fisheye distortion.

    Raw raycasting produces a "fisheye" effect because rays at the edges
    of the FOV travel further to hit walls at the same perpendicular distance.

    Correction: d_corrected = d_raw * cos(ray_angle - player_angle)

    Args:
        dists: array of shape (r,) — raw distances for each ray
        angles: array of shape (r,) — angle of each ray
        player_angle: scalar — player's facing direction

    Returns:
        Array of shape (r,) — corrected distances

    Example:
        A wall 10 units away perpendicular to the player:
        - Center ray (angle = player_angle): d = 10, correction = 10 * cos(0) = 10
        - Edge ray (30° off): d ≈ 11.5, correction = 11.5 * cos(30°) ≈ 10

    Hint:
        Element-wise multiply distances by cos of angle difference.
        This is simple — no einsum needed, just jnp.cos and multiplication.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Remove the distortion. See clearly.")


def distance_to_height(dists, screen_h):
    """
    Convert distances to wall heights on screen.

    Closer walls appear taller, distant walls appear shorter.
    Formula: height = screen_height / distance

    Args:
        dists: array of shape (r,) — distances to walls
        screen_h: int — screen height in pixels

    Returns:
        Array of shape (r,) — wall heights in pixels

    Example:
        >>> distance_to_height(jnp.array([1.0, 2.0, 4.0]), 480)
        [480.0, 240.0, 120.0]

    Hint:
        Simple division. Watch for zero distances (add small epsilon).
    """
    # YOUR CODE HERE
    raise NotImplementedError("Distance to height. Perspective projection.")


def shade_by_distance(colors, dists, max_dist):
    """
    Apply distance-based shading to colors (fog effect).

    Farther objects appear darker, simulating fog/atmosphere.
    shade_factor = 1 - (distance / max_distance), clamped to [0.2, 1.0]

    Args:
        colors: array of shape (r, 3) — RGB colors for each column
        dists: array of shape (r,) — distances for each column
        max_dist: scalar — distance at which shading reaches minimum

    Returns:
        Array of shape (r, 3) — shaded colors

    Example:
        >>> colors = jnp.array([[255, 0, 0], [0, 255, 0]])  # red, green
        >>> dists = jnp.array([5.0, 15.0])
        >>> shade_by_distance(colors, dists, 30.0)
        # Red is at 5/30 = 0.17 -> factor = 0.83
        # Green is at 15/30 = 0.5 -> factor = 0.5

    Hint:
        Compute shade factors from distances (one factor per ray).
        Then multiply each color's RGB components by its factor.
        How do you broadcast a per-ray scalar across the 3 color channels?
    """
    # YOUR CODE HERE
    raise NotImplementedError("Shade the world. Distance fades all.")


def build_column_masks(heights, screen_h):
    """
    Build a mask array for rendering wall columns.

    For each ray, we need to know which screen rows are part of the wall.
    Wall is centered vertically: rows from (screen_h - height)/2 to (screen_h + height)/2.

    Args:
        heights: array of shape (r,) — wall heights for each column
        screen_h: int — screen height in pixels

    Returns:
        Array of shape (screen_h, r) — boolean mask
            True where the wall should be drawn

    Example:
        >>> heights = jnp.array([100, 200])
        >>> build_column_masks(heights, 300)
        # For column 0: wall from row 100 to 200 (height 100, centered)
        # For column 1: wall from row 50 to 250 (height 200, centered)

    Hint:
        1. Compute top and bottom row for each column
        2. Create row indices: jnp.arange(screen_h).reshape(-1, 1)
        3. Compare: top <= row < bottom
        Use broadcasting, not einsum.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Build the mask. Define visibility.")
