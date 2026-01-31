"""
Tests for Chapter 7: Spooky Action at a Distance — Einstein Math

Run with: uv run pytest tests/test_c07_spooky_action_at_a_distance.py -v
"""

import math

import jax.numpy as jnp
import pytest

from solutions.c07_spooky_action_at_a_distance import (
    point_distances,
    all_pairs_distances,
    points_to_angles,
    angle_in_fov,
    project_to_screen_x,
    sprite_scale,
)


class TestPointDistances:
    """Tests for point_distances function."""

    def test_basic(self):
        origin = jnp.array([0.0, 0.0])
        points = jnp.array([[3.0, 4.0], [5.0, 12.0]])
        result = point_distances(origin, points)
        expected = jnp.array([5.0, 13.0])
        assert jnp.allclose(result, expected)

    def test_same_point(self):
        origin = jnp.array([5.0, 5.0])
        points = jnp.array([[5.0, 5.0]])
        result = point_distances(origin, points)
        assert jnp.isclose(result[0], 0.0)

    def test_unit_distances(self):
        origin = jnp.array([0.0, 0.0])
        points = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        result = point_distances(origin, points)
        expected = jnp.array([1.0, 1.0, 1.0, 1.0])
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        origin = jnp.array([0.0, 0.0])
        points = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        result = point_distances(origin, points)
        assert result.shape == (4,)


class TestAllPairsDistances:
    """Tests for all_pairs_distances function."""

    def test_basic(self):
        a = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        b = jnp.array([[0.0, 1.0], [1.0, 1.0]])
        result = all_pairs_distances(a, b)
        expected = jnp.array([
            [1.0, math.sqrt(2)],
            [math.sqrt(2), 1.0]
        ])
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_same_sets(self):
        points = jnp.array([[0.0, 0.0], [3.0, 4.0]])
        result = all_pairs_distances(points, points)
        # Diagonal should be zeros
        assert jnp.isclose(result[0, 0], 0.0)
        assert jnp.isclose(result[1, 1], 0.0)
        # Off-diagonal should be 5
        assert jnp.isclose(result[0, 1], 5.0)
        assert jnp.isclose(result[1, 0], 5.0)

    def test_output_shape(self):
        a = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])  # 3 points
        b = jnp.array([[0.0, 1.0], [1.0, 1.0]])  # 2 points
        result = all_pairs_distances(a, b)
        assert result.shape == (3, 2)


class TestPointsToAngles:
    """Tests for points_to_angles function."""

    def test_cardinal_directions(self):
        origin = jnp.array([0.0, 0.0])
        points = jnp.array([
            [1.0, 0.0],   # East
            [0.0, 1.0],   # North
            [-1.0, 0.0],  # West
            [0.0, -1.0],  # South
        ])
        result = points_to_angles(origin, points)
        expected = jnp.array([0.0, math.pi / 2, math.pi, -math.pi / 2])
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_diagonal_directions(self):
        origin = jnp.array([0.0, 0.0])
        points = jnp.array([
            [1.0, 1.0],   # Northeast
            [-1.0, 1.0],  # Northwest
        ])
        result = points_to_angles(origin, points)
        expected = jnp.array([math.pi / 4, 3 * math.pi / 4])
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_non_origin(self):
        origin = jnp.array([5.0, 5.0])
        points = jnp.array([[6.0, 5.0]])  # East of origin
        result = points_to_angles(origin, points)
        assert jnp.isclose(result[0], 0.0, atol=1e-5)


class TestAngleInFov:
    """Tests for angle_in_fov function."""

    def test_center_in_fov(self):
        angles = jnp.array([0.5])
        result = angle_in_fov(angles, player_angle=0.5, fov=1.0)
        assert result[0]

    def test_edge_in_fov(self):
        angles = jnp.array([0.0, 1.0])
        result = angle_in_fov(angles, player_angle=0.5, fov=1.0)
        assert result[0]  # Left edge
        assert result[1]  # Right edge

    def test_outside_fov(self):
        angles = jnp.array([-0.5, 1.5])
        result = angle_in_fov(angles, player_angle=0.5, fov=1.0)
        assert not result[0]  # Too far left
        assert not result[1]  # Too far right

    def test_wraparound(self):
        # Test angle wraparound at 2π boundary
        angles = jnp.array([0.1])
        # Player facing nearly the same direction (just wrapped around)
        result = angle_in_fov(angles, player_angle=2 * math.pi - 0.1, fov=1.0)
        assert result[0]  # Should be in FOV despite numerical difference


class TestProjectToScreenX:
    """Tests for project_to_screen_x function."""

    def test_center(self):
        angles = jnp.array([0.0])
        result = project_to_screen_x(angles, player_angle=0.0, fov=math.pi / 3, width=640)
        assert jnp.isclose(result[0], 320.0, atol=1.0)  # Center of screen

    def test_left_edge(self):
        fov = math.pi / 3  # 60 degrees
        angles = jnp.array([-fov / 2])
        result = project_to_screen_x(angles, player_angle=0.0, fov=fov, width=640)
        assert jnp.isclose(result[0], 0.0, atol=1.0)  # Left edge

    def test_right_edge(self):
        fov = math.pi / 3
        angles = jnp.array([fov / 2])
        result = project_to_screen_x(angles, player_angle=0.0, fov=fov, width=640)
        assert jnp.isclose(result[0], 640.0, atol=1.0)  # Right edge

    def test_output_shape(self):
        angles = jnp.array([0.0, 0.1, 0.2, 0.3])
        result = project_to_screen_x(angles, player_angle=0.15, fov=1.0, width=640)
        assert result.shape == (4,)


class TestSpriteScale:
    """Tests for sprite_scale function."""

    def test_basic(self):
        dists = jnp.array([1.0, 2.0, 4.0])
        result = sprite_scale(dists, base_size=100)
        expected = jnp.array([100.0, 50.0, 25.0])
        assert jnp.allclose(result, expected)

    def test_close_distance(self):
        dists = jnp.array([0.5])
        result = sprite_scale(dists, base_size=64)
        assert result[0] > 64  # Closer = bigger

    def test_far_distance(self):
        dists = jnp.array([100.0])
        result = sprite_scale(dists, base_size=64)
        assert result[0] < 1  # Far = tiny

    def test_output_shape(self):
        dists = jnp.array([5.0, 10.0, 15.0, 20.0])
        result = sprite_scale(dists, base_size=64)
        assert result.shape == (4,)
