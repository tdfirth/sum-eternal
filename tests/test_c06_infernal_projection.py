"""
Tests for Chapter 6: Infernal Projection — Screen Rendering

Run with: uv run pytest tests/test_c06_infernal_projection.py -v
"""

import math

import jax.numpy as jnp
import pytest

from solutions.c06_infernal_projection import (
    fisheye_correct,
    distance_to_height,
    shade_by_distance,
    build_column_masks,
)


class TestFisheyeCorrect:
    """Tests for fisheye_correct function."""

    def test_center_ray_unchanged(self):
        dists = jnp.array([10.0])
        angles = jnp.array([0.5])
        player_angle = 0.5
        result = fisheye_correct(dists, angles, player_angle)
        # cos(0) = 1, so distance unchanged
        assert jnp.isclose(result[0], 10.0)

    def test_edge_rays_corrected(self):
        dists = jnp.array([10.0, 10.0])
        player_angle = 0.0
        angles = jnp.array([math.pi / 6, -math.pi / 6])  # 30 degrees off center
        result = fisheye_correct(dists, angles, player_angle)
        expected = 10.0 * math.cos(math.pi / 6)  # ~8.66
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_preserves_shape(self):
        dists = jnp.array([5.0, 10.0, 15.0, 20.0])
        angles = jnp.array([0.0, 0.1, 0.2, 0.3])
        result = fisheye_correct(dists, angles, 0.15)
        assert result.shape == (4,)


class TestDistanceToHeight:
    """Tests for distance_to_height function."""

    def test_basic(self):
        dists = jnp.array([1.0, 2.0, 4.0])
        result = distance_to_height(dists, 480)
        expected = jnp.array([480.0, 240.0, 120.0])
        assert jnp.allclose(result, expected)

    def test_close_distance(self):
        dists = jnp.array([0.5])
        result = distance_to_height(dists, 480)
        assert result[0] > 480  # Closer = taller

    def test_far_distance(self):
        dists = jnp.array([100.0])
        result = distance_to_height(dists, 480)
        assert result[0] < 10  # Far = short

    def test_preserves_shape(self):
        dists = jnp.array([5.0, 10.0, 15.0])
        result = distance_to_height(dists, 480)
        assert result.shape == (3,)


class TestShadeByDistance:
    """Tests for shade_by_distance function."""

    def test_close_objects_bright(self):
        colors = jnp.array([[255.0, 255.0, 255.0]])
        dists = jnp.array([0.0])
        result = shade_by_distance(colors, dists, 30.0)
        # At distance 0, shade factor should be 1.0
        assert jnp.allclose(result, colors, atol=1.0)

    def test_far_objects_dark(self):
        colors = jnp.array([[255.0, 255.0, 255.0]])
        dists = jnp.array([30.0])
        result = shade_by_distance(colors, dists, 30.0)
        # At max distance, shade factor should be minimum (0.2)
        expected = jnp.array([[51.0, 51.0, 51.0]])  # 255 * 0.2
        assert jnp.allclose(result, expected, atol=5.0)

    def test_preserves_color_ratios(self):
        colors = jnp.array([[200.0, 100.0, 50.0]])
        dists = jnp.array([15.0])
        result = shade_by_distance(colors, dists, 30.0)
        # All channels should be scaled by same factor
        # Ratio should be preserved
        assert jnp.isclose(result[0, 0] / result[0, 1], 2.0, atol=0.1)
        assert jnp.isclose(result[0, 1] / result[0, 2], 2.0, atol=0.1)

    def test_output_shape(self):
        colors = jnp.array([[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0]])
        dists = jnp.array([5.0, 10.0, 15.0])
        result = shade_by_distance(colors, dists, 30.0)
        assert result.shape == (3, 3)


class TestBuildColumnMasks:
    """Tests for build_column_masks function."""

    def test_basic_shape(self):
        heights = jnp.array([100.0, 200.0, 150.0])
        result = build_column_masks(heights, 300)
        assert result.shape == (300, 3)

    def test_wall_centered(self):
        heights = jnp.array([100.0])
        result = build_column_masks(heights, 300)
        # Wall of height 100 should span rows 100-200 (centered)
        # Rows 0-99 should be False (ceiling)
        # Rows 100-199 should be True (wall)
        # Rows 200-299 should be False (floor)
        assert not result[50, 0]   # Ceiling
        assert result[150, 0]      # Wall
        assert not result[250, 0]  # Floor

    def test_full_height_wall(self):
        heights = jnp.array([300.0])
        result = build_column_masks(heights, 300)
        # Wall fills entire screen
        assert jnp.all(result[:, 0])

    def test_tiny_wall(self):
        heights = jnp.array([10.0])
        result = build_column_masks(heights, 300)
        # Most rows should be False
        true_count = jnp.sum(result[:, 0])
        assert true_count <= 20  # Small wall

    def test_multiple_columns(self):
        heights = jnp.array([50.0, 150.0, 250.0])
        result = build_column_masks(heights, 300)
        # Taller walls should have more True values
        counts = jnp.sum(result, axis=0)
        assert counts[0] < counts[1] < counts[2]
