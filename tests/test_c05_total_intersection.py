"""
Tests for Chapter 5: Total Intersection — Ray-Wall Math

Run with: uv run pytest tests/test_c05_total_intersection.py -v
"""

import math

import jax.numpy as jnp
import pytest

from solutions.c05_total_intersection import (
    cross_2d,
    batch_cross_2d,
    all_pairs_cross_2d,
    ray_wall_determinants,
    ray_wall_t_values,
    ray_wall_s_values,
)


class TestCross2D:
    """Tests for cross_2d function."""

    def test_perpendicular_vectors(self):
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 1.0])
        result = cross_2d(a, b)
        assert jnp.isclose(result, 1.0)

    def test_perpendicular_opposite(self):
        a = jnp.array([0.0, 1.0])
        b = jnp.array([1.0, 0.0])
        result = cross_2d(a, b)
        assert jnp.isclose(result, -1.0)

    def test_parallel_vectors(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([2.0, 4.0])
        result = cross_2d(a, b)
        assert jnp.isclose(result, 0.0)

    def test_general_vectors(self):
        a = jnp.array([3.0, 4.0])
        b = jnp.array([1.0, 2.0])
        result = cross_2d(a, b)
        # 3*2 - 4*1 = 2
        assert jnp.isclose(result, 2.0)


class TestBatchCross2D:
    """Tests for batch_cross_2d function."""

    def test_basic(self):
        a = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        result = batch_cross_2d(a, b)
        expected = jnp.array([1.0, -1.0])
        assert jnp.allclose(result, expected)

    def test_parallel_pairs(self):
        a = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        result = batch_cross_2d(a, b)
        expected = jnp.array([0.0, 0.0])
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        a = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        result = batch_cross_2d(a, b)
        assert result.shape == (3,)


class TestAllPairsCross2D:
    """Tests for all_pairs_cross_2d function."""

    def test_basic(self):
        a = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b = jnp.array([[0.0, 1.0], [1.0, 1.0]])
        result = all_pairs_cross_2d(a, b)
        # a[0] x b[0] = 1*1 - 0*0 = 1
        # a[0] x b[1] = 1*1 - 0*1 = 1
        # a[1] x b[0] = 0*1 - 1*0 = 0  -- wait, should be: 0*1 - 1*0 = 0
        # Actually: a[1] x b[0] = 0*1 - 1*0 = 0  No wait...
        # cross(a, b) = a[0]*b[1] - a[1]*b[0]
        # a[1] x b[0] = 0*1 - 1*0 = 0
        # a[1] x b[1] = 0*1 - 1*1 = -1
        expected = jnp.array([[1.0, 1.0], [0.0, -1.0]])
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        a = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3 rays
        b = jnp.array([[0.0, 1.0], [1.0, 0.0]])  # 2 walls
        result = all_pairs_cross_2d(a, b)
        assert result.shape == (3, 2)


class TestRayWallDeterminants:
    """Tests for ray_wall_determinants function."""

    def test_perpendicular_intersection(self):
        ray_dirs = jnp.array([[1.0, 0.0]])  # Ray pointing east
        wall_dirs = jnp.array([[0.0, 1.0]])  # Wall pointing north
        result = ray_wall_determinants(ray_dirs, wall_dirs)
        assert jnp.isclose(result[0, 0], 1.0)

    def test_parallel_no_intersection(self):
        ray_dirs = jnp.array([[1.0, 0.0]])  # Ray pointing east
        wall_dirs = jnp.array([[2.0, 0.0]])  # Wall also pointing east
        result = ray_wall_determinants(ray_dirs, wall_dirs)
        assert jnp.isclose(result[0, 0], 0.0)

    def test_multiple_rays_walls(self):
        ray_dirs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        wall_dirs = jnp.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
        result = ray_wall_determinants(ray_dirs, wall_dirs)
        assert result.shape == (2, 3)


class TestRayWallTValues:
    """Tests for ray_wall_t_values function."""

    def test_simple_intersection(self):
        # Ray from origin pointing east
        player = jnp.array([0.0, 0.0])
        ray_dirs = jnp.array([[1.0, 0.0]])
        # Vertical wall at x=5, from y=-10 to y=10
        wall_starts = jnp.array([[5.0, -10.0]])
        wall_dirs = jnp.array([[0.0, 20.0]])

        result = ray_wall_t_values(player, ray_dirs, wall_starts, wall_dirs)
        assert jnp.isclose(result[0, 0], 5.0, atol=1e-5)

    def test_ray_behind_wall(self):
        # Ray from (10, 0) pointing east
        player = jnp.array([10.0, 0.0])
        ray_dirs = jnp.array([[1.0, 0.0]])
        # Wall at x=5
        wall_starts = jnp.array([[5.0, -10.0]])
        wall_dirs = jnp.array([[0.0, 20.0]])

        result = ray_wall_t_values(player, ray_dirs, wall_starts, wall_dirs)
        # t should be negative (wall is behind)
        assert result[0, 0] < 0

    def test_output_shape(self):
        player = jnp.array([0.0, 0.0])
        ray_dirs = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        wall_starts = jnp.array([[5.0, 0.0], [0.0, 5.0]])
        wall_dirs = jnp.array([[0.0, 10.0], [10.0, 0.0]])

        result = ray_wall_t_values(player, ray_dirs, wall_starts, wall_dirs)
        assert result.shape == (3, 2)


class TestRayWallSValues:
    """Tests for ray_wall_s_values function."""

    def test_hits_middle_of_wall(self):
        # Ray from origin pointing northeast, wall along x=5
        player = jnp.array([0.0, 0.0])
        ray_dirs = jnp.array([[1.0, 0.0]])
        wall_starts = jnp.array([[5.0, -5.0]])
        wall_dirs = jnp.array([[0.0, 10.0]])  # Wall from y=-5 to y=5

        result = ray_wall_s_values(player, ray_dirs, wall_starts, wall_dirs)
        # Ray hits at y=0, which is at s=0.5 along the wall
        assert jnp.isclose(result[0, 0], 0.5, atol=1e-5)

    def test_hits_start_of_wall(self):
        player = jnp.array([0.0, -5.0])
        ray_dirs = jnp.array([[1.0, 0.0]])
        wall_starts = jnp.array([[5.0, -5.0]])
        wall_dirs = jnp.array([[0.0, 10.0]])

        result = ray_wall_s_values(player, ray_dirs, wall_starts, wall_dirs)
        assert jnp.isclose(result[0, 0], 0.0, atol=1e-5)

    def test_misses_wall(self):
        # Ray pointing away from wall
        player = jnp.array([0.0, 0.0])
        ray_dirs = jnp.array([[1.0, 0.0]])
        wall_starts = jnp.array([[5.0, 10.0]])  # Wall starts above ray path
        wall_dirs = jnp.array([[0.0, 10.0]])    # Extends further up

        result = ray_wall_s_values(player, ray_dirs, wall_starts, wall_dirs)
        # s should be negative (intersection point is before wall start)
        assert result[0, 0] < 0

    def test_output_shape(self):
        player = jnp.array([0.0, 0.0])
        ray_dirs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        wall_starts = jnp.array([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0]])
        wall_dirs = jnp.array([[0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])

        result = ray_wall_s_values(player, ray_dirs, wall_starts, wall_dirs)
        assert result.shape == (2, 3)
