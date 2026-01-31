"""
Chapter 9: Nightmare Mode
=========================
Advanced challenges — for those who seek true mastery.

Optional. The notation has depths you haven't seen.
Textures. Interpolation. The full visual experience.

Complete these functions using jax.numpy.einsum to proceed.
Run tests with: uv run pytest tests/test_c09_nightmare_mode.py -v
"""

import jax.numpy as jnp


def texture_column_lookup(hit_s, tex_width):
    """
    Convert wall hit s-values to texture column indices.

    When a ray hits a wall at position s along the wall (0 to 1),
    we need to know which column of the texture to sample.

    Args:
        hit_s: array of shape (r,) — s values where rays hit walls [0, 1]
        tex_width: int — width of texture in pixels

    Returns:
        Array of shape (r,) — texture column indices [0, tex_width-1]

    Example:
        >>> hit_s = jnp.array([0.0, 0.5, 0.999])
        >>> texture_column_lookup(hit_s, 64)
        [0, 32, 63]

    Hint:
        col = floor(s * tex_width), clamped to [0, tex_width-1]
        Handle edge case where s = 1.0 exactly.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Map the hit position to texture coordinates.")


def bilinear_sample(texture, coords):
    """
    Sample a texture using bilinear interpolation.

    Given floating-point coordinates, interpolate between the four
    nearest texels (texture pixels) for smooth sampling.

    Args:
        texture: array of shape (h, w, 3) — RGB texture
        coords: array of shape (n, 2) — (y, x) coordinates in [0, h-1] x [0, w-1]

    Returns:
        Array of shape (n, 3) — interpolated colors

    Example:
        >>> texture = jnp.array([[[255, 0, 0], [0, 255, 0]],
        ...                      [[0, 0, 255], [255, 255, 0]]])  # 2x2
        >>> coords = jnp.array([[0.5, 0.5]])  # center of texture
        >>> bilinear_sample(texture, coords)
        [[127.5, 127.5, 127.5]]  # average of all four corners

    Hint:
        1. Split coords into integer and fractional parts
        2. Get the four corner texel values
        3. Linearly interpolate in x, then in y (or vice versa)

        For the interpolation weights, use einsum or broadcasting carefully.
        This is one of the harder functions — take it step by step.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Bilinear interpolation. Smooth sampling.")


def floor_cast_coords(screen_y, player_pos, player_angle):
    """
    Compute world coordinates for floor casting.

    For each row of pixels below the horizon, compute where in the world
    that row corresponds to. This enables textured floors.

    Args:
        screen_y: array of shape (num_rows,) — y coordinates of floor rows
                  (from horizon to bottom of screen)
        player_pos: array of shape (2,) — player position in world
        player_angle: scalar — player facing angle

    Returns:
        Array of shape (num_rows, screen_width, 2) — world (x, y) for each pixel
            (Assume screen_width is derived from FOV and some standard)

    Note:
        This function is complex. It involves:
        1. Computing distance to floor at each row (based on row distance from horizon)
        2. Computing the horizontal position across the row
        3. Transforming to world coordinates using player position and angle

    Hint:
        For row at screen position y below horizon:
        - row_distance = screen_h / (2 * (y - horizon_y) + 1)
          (This is an approximation; exact formula depends on perspective)

        For each pixel in that row:
        - Compute the angle (left edge to right edge of FOV)
        - Compute world position: player_pos + row_distance * direction

        This involves trigonometry and careful coordinate mapping.
        Use einsum where it helps with batched operations.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Floor casting. The ground beneath your feet.")
