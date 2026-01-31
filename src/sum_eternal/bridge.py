"""
Bridge - Connects student solutions to the game engine.

Handles NotImplementedError gracefully, enabling progressive game states.
Provides safe wrappers around student functions and progress detection.
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import Callable, Any
import math

import jax.numpy as jnp


class Progress(IntEnum):
    """Tracks how far the student has progressed.

    Each level represents completion of a chapter's functions.
    The game renders differently based on the progress level.
    """
    NOTHING = 0
    CHAPTER_1_COMPLETE = auto()  # Basic ops - debug view
    CHAPTER_2_COMPLETE = auto()  # Matrix ops - 2D map appears
    CHAPTER_3_COMPLETE = auto()  # Batch ops - player + rays
    CHAPTER_4_COMPLETE = auto()  # Ray generation - ray fan visible
    CHAPTER_5_COMPLETE = auto()  # Ray-wall intersection - 3D unlocked!
    CHAPTER_6_COMPLETE = auto()  # Projection/shading - full rendering
    CHAPTER_7_COMPLETE = auto()  # Einstein math - Einsteins visible
    CHAPTER_8_COMPLETE = auto()  # Combat - game complete
    CHAPTER_9_COMPLETE = auto()  # Nightmare mode - textures


def _safe_call(func: Callable, *args, **kwargs) -> Any | None:
    """Call a function, returning None if it raises NotImplementedError or fails."""
    try:
        result = func(*args, **kwargs)
        if result is None:
            return None
        return result
    except NotImplementedError:
        return None
    except Exception:
        return None


def check_progress() -> Progress:
    """Determine current progress by checking which functions are implemented.

    Runs minimal smoke tests to verify implementations work.
    Returns the highest chapter where all functions pass.
    """
    # Chapter 1: Basic contractions
    try:
        from solutions.c01_first_blood import (
            vector_sum, element_multiply, dot_product,
            outer_product, matrix_vector_mul, matrix_matrix_mul
        )

        v = jnp.array([1.0, 2.0, 3.0])
        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0, 4.0])
        M = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        if _safe_call(vector_sum, v) is None:
            return Progress.NOTHING
        if _safe_call(element_multiply, a, b) is None:
            return Progress.NOTHING
        if _safe_call(dot_product, a, b) is None:
            return Progress.NOTHING
        if _safe_call(outer_product, a, b) is None:
            return Progress.NOTHING
        if _safe_call(matrix_vector_mul, M, a) is None:
            return Progress.NOTHING
        if _safe_call(matrix_matrix_mul, M, M) is None:
            return Progress.NOTHING
    except (ImportError, Exception):
        return Progress.NOTHING

    # Chapter 2: Reductions & rearrangements
    try:
        from solutions.c02_knee_deep_in_the_indices import (
            transpose, trace, diag_extract,
            sum_rows, sum_cols, frobenius_norm_sq
        )

        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        if _safe_call(transpose, M) is None:
            return Progress.CHAPTER_1_COMPLETE
        if _safe_call(trace, M) is None:
            return Progress.CHAPTER_1_COMPLETE
        if _safe_call(diag_extract, M) is None:
            return Progress.CHAPTER_1_COMPLETE
        if _safe_call(sum_rows, M) is None:
            return Progress.CHAPTER_1_COMPLETE
        if _safe_call(sum_cols, M) is None:
            return Progress.CHAPTER_1_COMPLETE
        if _safe_call(frobenius_norm_sq, M) is None:
            return Progress.CHAPTER_1_COMPLETE
    except (ImportError, Exception):
        return Progress.CHAPTER_1_COMPLETE

    # Chapter 3: Batch operations
    try:
        from solutions.c03_the_slaughter_batch import (
            batch_vector_sum, batch_dot_pairwise, batch_magnitude_sq,
            all_pairs_dot, batch_matrix_vector, batch_outer
        )

        batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        M = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        if _safe_call(batch_vector_sum, batch) is None:
            return Progress.CHAPTER_2_COMPLETE
        if _safe_call(batch_dot_pairwise, batch, batch) is None:
            return Progress.CHAPTER_2_COMPLETE
        if _safe_call(batch_magnitude_sq, batch) is None:
            return Progress.CHAPTER_2_COMPLETE
        if _safe_call(all_pairs_dot, batch, batch) is None:
            return Progress.CHAPTER_2_COMPLETE
        if _safe_call(batch_matrix_vector, M, batch) is None:
            return Progress.CHAPTER_2_COMPLETE
        if _safe_call(batch_outer, batch, batch) is None:
            return Progress.CHAPTER_2_COMPLETE
    except (ImportError, Exception):
        return Progress.CHAPTER_2_COMPLETE

    # Chapter 4: Ray generation
    try:
        from solutions.c04_rip_and_trace import (
            angles_to_directions, rotate_vectors,
            normalize_vectors, scale_vectors
        )

        angles = jnp.array([0.0, math.pi / 2])
        vecs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        scales = jnp.array([2.0, 3.0])

        if _safe_call(angles_to_directions, angles) is None:
            return Progress.CHAPTER_3_COMPLETE
        if _safe_call(rotate_vectors, vecs, 0.5) is None:
            return Progress.CHAPTER_3_COMPLETE
        if _safe_call(normalize_vectors, vecs) is None:
            return Progress.CHAPTER_3_COMPLETE
        if _safe_call(scale_vectors, vecs, scales) is None:
            return Progress.CHAPTER_3_COMPLETE
    except (ImportError, Exception):
        return Progress.CHAPTER_3_COMPLETE

    # Chapter 5: Ray-wall intersection
    try:
        from solutions.c05_total_intersection import (
            cross_2d, batch_cross_2d, all_pairs_cross_2d,
            ray_wall_determinants, ray_wall_t_values, ray_wall_s_values
        )

        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 1.0])
        batch_a = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        batch_b = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        player = jnp.array([0.0, 0.0])

        if _safe_call(cross_2d, a, b) is None:
            return Progress.CHAPTER_4_COMPLETE
        if _safe_call(batch_cross_2d, batch_a, batch_b) is None:
            return Progress.CHAPTER_4_COMPLETE
        if _safe_call(all_pairs_cross_2d, batch_a, batch_b) is None:
            return Progress.CHAPTER_4_COMPLETE
        if _safe_call(ray_wall_determinants, batch_a, batch_b) is None:
            return Progress.CHAPTER_4_COMPLETE
        if _safe_call(ray_wall_t_values, player, batch_a, batch_b, batch_a) is None:
            return Progress.CHAPTER_4_COMPLETE
        if _safe_call(ray_wall_s_values, player, batch_a, batch_b, batch_a) is None:
            return Progress.CHAPTER_4_COMPLETE
    except (ImportError, Exception):
        return Progress.CHAPTER_4_COMPLETE

    # Chapter 6: Projection/shading
    try:
        from solutions.c06_infernal_projection import (
            fisheye_correct, distance_to_height,
            shade_by_distance, build_column_masks
        )

        dists = jnp.array([5.0, 10.0, 15.0])
        angles = jnp.array([0.0, 0.1, 0.2])
        colors = jnp.array([[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0]])
        heights = jnp.array([100.0, 200.0, 150.0])

        if _safe_call(fisheye_correct, dists, angles, 0.0) is None:
            return Progress.CHAPTER_5_COMPLETE
        if _safe_call(distance_to_height, dists, 480) is None:
            return Progress.CHAPTER_5_COMPLETE
        if _safe_call(shade_by_distance, colors, dists, 30.0) is None:
            return Progress.CHAPTER_5_COMPLETE
        if _safe_call(build_column_masks, heights, 480) is None:
            return Progress.CHAPTER_5_COMPLETE
    except (ImportError, Exception):
        return Progress.CHAPTER_5_COMPLETE

    # Chapter 7: Einstein math
    try:
        from solutions.c07_spooky_action_at_a_distance import (
            point_distances, all_pairs_distances, points_to_angles,
            angle_in_fov, project_to_screen_x, sprite_scale
        )

        origin = jnp.array([0.0, 0.0])
        points = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        if _safe_call(point_distances, origin, points) is None:
            return Progress.CHAPTER_6_COMPLETE
        if _safe_call(all_pairs_distances, points, points) is None:
            return Progress.CHAPTER_6_COMPLETE
        if _safe_call(points_to_angles, origin, points) is None:
            return Progress.CHAPTER_6_COMPLETE
        if _safe_call(angle_in_fov, jnp.array([0.0, 0.5]), 0.0, 1.0) is None:
            return Progress.CHAPTER_6_COMPLETE
        if _safe_call(project_to_screen_x, jnp.array([0.0, 0.5]), 0.0, 1.0, 640) is None:
            return Progress.CHAPTER_6_COMPLETE
        if _safe_call(sprite_scale, jnp.array([5.0, 10.0]), 64) is None:
            return Progress.CHAPTER_6_COMPLETE
    except (ImportError, Exception):
        return Progress.CHAPTER_6_COMPLETE

    # Chapter 8: Combat
    try:
        from solutions.c08_the_icon_of_ein import (
            project_points_onto_ray, perpendicular_distance_to_ray,
            ray_hits_target, move_toward_point
        )

        points = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        origin = jnp.array([0.0, 0.0])
        direction = jnp.array([1.0, 0.0])
        positions = jnp.array([[5.0, 5.0], [10.0, 10.0]])
        target = jnp.array([0.0, 0.0])
        speeds = jnp.array([1.0, 2.0])

        if _safe_call(project_points_onto_ray, points, origin, direction) is None:
            return Progress.CHAPTER_7_COMPLETE
        if _safe_call(perpendicular_distance_to_ray, points, origin, direction) is None:
            return Progress.CHAPTER_7_COMPLETE
        if _safe_call(ray_hits_target, jnp.array([1.0, 2.0]), jnp.array([0.1, 0.2]), jnp.array([0.5, 0.5]), 10.0) is None:
            return Progress.CHAPTER_7_COMPLETE
        if _safe_call(move_toward_point, positions, target, speeds, 0.1) is None:
            return Progress.CHAPTER_7_COMPLETE
    except (ImportError, Exception):
        return Progress.CHAPTER_7_COMPLETE

    # Chapter 9: Nightmare mode (optional)
    try:
        from solutions.c09_nightmare_mode import (
            texture_column_lookup, bilinear_sample, floor_cast_coords
        )

        hit_s = jnp.array([0.0, 0.5, 1.0])
        texture = jnp.ones((64, 64, 3))
        coords = jnp.array([[0.5, 0.5], [0.25, 0.75]])

        if _safe_call(texture_column_lookup, hit_s, 64) is None:
            return Progress.CHAPTER_8_COMPLETE
        if _safe_call(bilinear_sample, texture, coords) is None:
            return Progress.CHAPTER_8_COMPLETE
        if _safe_call(floor_cast_coords, jnp.arange(240, 480), jnp.array([0.0, 0.0]), 0.0) is None:
            return Progress.CHAPTER_8_COMPLETE

        return Progress.CHAPTER_9_COMPLETE
    except (ImportError, Exception):
        return Progress.CHAPTER_8_COMPLETE


def has_raycasting_functions() -> bool:
    """Check if all raycasting functions are available."""
    return check_progress() >= Progress.CHAPTER_5_COMPLETE


def cast_all_rays(
    player_pos: tuple[float, float],
    player_angle: float,
    fov: float,
    num_rays: int,
    wall_data: dict
) -> tuple[list[float], list[tuple[int, int, int]]]:
    """Cast all rays and return distances and wall colors.

    This function uses the student's implementations from Chapters 4-5.

    Args:
        player_pos: (x, y) player position
        player_angle: Player facing angle in radians
        fov: Field of view in radians
        num_rays: Number of rays to cast
        wall_data: Wall data from Map.wall_data

    Returns:
        Tuple of (distances, colors) where:
            distances: List of distances to nearest wall for each ray
            colors: List of (r, g, b) colors for each ray's wall hit
    """
    try:
        from solutions.c04_rip_and_trace import angles_to_directions
        from solutions.c05_total_intersection import (
            ray_wall_t_values, ray_wall_s_values
        )

        # Generate ray angles
        ray_angles = jnp.linspace(
            player_angle - fov / 2,
            player_angle + fov / 2,
            num_rays
        )

        # Convert angles to direction vectors
        ray_dirs = angles_to_directions(ray_angles)

        # Get wall data
        wall_starts = wall_data["wall_starts"]
        wall_dirs = wall_data["wall_dirs"]
        wall_colors = wall_data["wall_colors"]
        num_walls = wall_data["num_walls"]

        # Compute ray-wall intersections
        player = jnp.array(player_pos)
        t_values = ray_wall_t_values(player, ray_dirs, wall_starts, wall_dirs)
        s_values = ray_wall_s_values(player, ray_dirs, wall_starts, wall_dirs)

        # Filter valid intersections (t > 0, 0 <= s <= 1)
        valid = (t_values > 0.001) & (s_values >= 0) & (s_values <= 1)
        t_values = jnp.where(valid, t_values, jnp.inf)

        # Find nearest wall for each ray
        nearest_wall_idx = jnp.argmin(t_values, axis=1)
        distances = t_values[jnp.arange(num_rays), nearest_wall_idx]

        # Get colors for each ray
        colors = [
            wall_colors[int(idx)] if dist < jnp.inf else (64, 64, 64)
            for idx, dist in zip(nearest_wall_idx, distances)
        ]

        return list(float(d) for d in distances), colors

    except Exception as e:
        # Return fallback if raycasting fails
        return [10.0] * num_rays, [(100, 100, 100)] * num_rays
