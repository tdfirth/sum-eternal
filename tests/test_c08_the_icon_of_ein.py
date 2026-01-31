"""
Tests for Chapter 8: The Icon of Ein — Combat

Run with: uv run pytest tests/test_c08_the_icon_of_ein.py -v
"""

import math

import jax.numpy as jnp
import pytest

from solutions.c08_the_icon_of_ein import (
    project_points_onto_ray,
    perpendicular_distance_to_ray,
    ray_hits_target,
    move_toward_point,
)


class TestProjectPointsOntoRay:
    """Tests for project_points_onto_ray function."""

    def test_basic(self):
        points = jnp.array([[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        origin = jnp.array([0.0, 0.0])
        direction = jnp.array([1.0, 0.0])  # Pointing east
        result = project_points_onto_ray(points, origin, direction)
        expected = jnp.array([2.0, 0.0, -1.0])
        assert jnp.allclose(result, expected)

    def test_diagonal_ray(self):
        points = jnp.array([[1.0, 1.0], [2.0, 2.0]])
        origin = jnp.array([0.0, 0.0])
        direction = jnp.array([1.0, 1.0]) / math.sqrt(2)  # Normalized northeast
        result = project_points_onto_ray(points, origin, direction)
        expected = jnp.array([math.sqrt(2), 2 * math.sqrt(2)])
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_non_origin(self):
        points = jnp.array([[5.0, 3.0]])
        origin = jnp.array([3.0, 3.0])
        direction = jnp.array([1.0, 0.0])
        result = project_points_onto_ray(points, origin, direction)
        assert jnp.isclose(result[0], 2.0)  # 5 - 3 = 2

    def test_output_shape(self):
        points = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        origin = jnp.array([0.0, 0.0])
        direction = jnp.array([1.0, 0.0])
        result = project_points_onto_ray(points, origin, direction)
        assert result.shape == (4,)


class TestPerpendicularDistanceToRay:
    """Tests for perpendicular_distance_to_ray function."""

    def test_points_on_ray(self):
        points = jnp.array([[1.0, 0.0], [5.0, 0.0]])
        origin = jnp.array([0.0, 0.0])
        direction = jnp.array([1.0, 0.0])
        result = perpendicular_distance_to_ray(points, origin, direction)
        expected = jnp.array([0.0, 0.0])
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_points_off_ray(self):
        points = jnp.array([[1.0, 1.0], [2.0, 3.0]])
        origin = jnp.array([0.0, 0.0])
        direction = jnp.array([1.0, 0.0])
        result = perpendicular_distance_to_ray(points, origin, direction)
        expected = jnp.array([1.0, 3.0])  # y-coordinates are perpendicular distances
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_diagonal_ray(self):
        points = jnp.array([[1.0, 0.0]])
        origin = jnp.array([0.0, 0.0])
        direction = jnp.array([1.0, 1.0]) / math.sqrt(2)  # Northeast
        result = perpendicular_distance_to_ray(points, origin, direction)
        # Distance from (1,0) to line y=x is 1/sqrt(2)
        expected = jnp.array([1.0 / math.sqrt(2)])
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_always_positive(self):
        points = jnp.array([[1.0, -2.0], [1.0, 2.0]])
        origin = jnp.array([0.0, 0.0])
        direction = jnp.array([1.0, 0.0])
        result = perpendicular_distance_to_ray(points, origin, direction)
        assert jnp.all(result >= 0)


class TestRayHitsTarget:
    """Tests for ray_hits_target function."""

    def test_hit(self):
        proj_dist = jnp.array([5.0])
        perp_dist = jnp.array([0.5])
        radii = jnp.array([1.0])
        wall_dist = 10.0
        result = ray_hits_target(proj_dist, perp_dist, radii, wall_dist)
        assert result[0]  # Should hit

    def test_miss_behind_player(self):
        proj_dist = jnp.array([-5.0])  # Behind player
        perp_dist = jnp.array([0.5])
        radii = jnp.array([1.0])
        wall_dist = 10.0
        result = ray_hits_target(proj_dist, perp_dist, radii, wall_dist)
        assert not result[0]

    def test_miss_behind_wall(self):
        proj_dist = jnp.array([15.0])  # Beyond wall
        perp_dist = jnp.array([0.5])
        radii = jnp.array([1.0])
        wall_dist = 10.0
        result = ray_hits_target(proj_dist, perp_dist, radii, wall_dist)
        assert not result[0]

    def test_miss_too_far_off_axis(self):
        proj_dist = jnp.array([5.0])
        perp_dist = jnp.array([2.0])  # Too far from ray
        radii = jnp.array([1.0])
        wall_dist = 10.0
        result = ray_hits_target(proj_dist, perp_dist, radii, wall_dist)
        assert not result[0]

    def test_multiple_targets(self):
        proj_dist = jnp.array([5.0, 15.0, 3.0, 5.0])
        perp_dist = jnp.array([0.5, 0.2, 2.0, 0.5])
        radii = jnp.array([1.0, 1.0, 1.0, 0.3])
        wall_dist = 10.0
        result = ray_hits_target(proj_dist, perp_dist, radii, wall_dist)
        expected = jnp.array([True, False, False, False])
        assert jnp.all(result == expected)


class TestMoveTowardPoint:
    """Tests for move_toward_point function."""

    def test_basic(self):
        positions = jnp.array([[10.0, 0.0], [0.0, 10.0]])
        target = jnp.array([0.0, 0.0])
        speeds = jnp.array([1.0, 2.0])
        dt = 1.0
        result = move_toward_point(positions, target, speeds, dt)
        # Should move 1 and 2 units toward origin respectively
        expected = jnp.array([[9.0, 0.0], [0.0, 8.0]])
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_diagonal_movement(self):
        positions = jnp.array([[10.0, 10.0]])
        target = jnp.array([0.0, 0.0])
        speeds = jnp.array([math.sqrt(2)])
        dt = 1.0
        result = move_toward_point(positions, target, speeds, dt)
        # Should move sqrt(2) units diagonally = 1 unit in each axis
        expected = jnp.array([[9.0, 9.0]])
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_time_step_scaling(self):
        positions = jnp.array([[10.0, 0.0]])
        target = jnp.array([0.0, 0.0])
        speeds = jnp.array([2.0])
        result_1 = move_toward_point(positions, target, speeds, dt=0.5)
        result_2 = move_toward_point(positions, target, speeds, dt=1.0)
        # dt=0.5 should move half as far as dt=1.0
        assert jnp.isclose(result_1[0, 0], 9.0)  # 10 - 2*0.5
        assert jnp.isclose(result_2[0, 0], 8.0)  # 10 - 2*1.0

    def test_output_shape(self):
        positions = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        target = jnp.array([0.0, 0.0])
        speeds = jnp.array([1.0, 2.0, 3.0])
        result = move_toward_point(positions, target, speeds, dt=0.1)
        assert result.shape == (3, 2)
