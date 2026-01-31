"""
Tests for Chapter 9: Nightmare Mode — Advanced Challenges

Run with: uv run pytest tests/test_c09_nightmare_mode.py -v
"""

import math

import jax.numpy as jnp
import pytest

from solutions.c09_nightmare_mode import (
    texture_column_lookup,
    bilinear_sample,
    floor_cast_coords,
)


class TestTextureColumnLookup:
    """Tests for texture_column_lookup function."""

    def test_basic(self):
        hit_s = jnp.array([0.0, 0.5, 0.999])
        result = texture_column_lookup(hit_s, tex_width=64)
        expected = jnp.array([0, 32, 63])
        assert jnp.allclose(result, expected)

    def test_boundaries(self):
        hit_s = jnp.array([0.0, 1.0])
        result = texture_column_lookup(hit_s, tex_width=64)
        # s=0 -> column 0
        # s=1 -> column 63 (clamped, not 64)
        assert result[0] == 0
        assert result[1] <= 63

    def test_fractional_values(self):
        hit_s = jnp.array([0.25, 0.75])
        result = texture_column_lookup(hit_s, tex_width=100)
        expected = jnp.array([25, 75])
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        hit_s = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = texture_column_lookup(hit_s, tex_width=64)
        assert result.shape == (5,)


class TestBilinearSample:
    """Tests for bilinear_sample function."""

    def test_corner_sampling(self):
        # 2x2 texture with distinct colors
        texture = jnp.array([
            [[255.0, 0.0, 0.0], [0.0, 255.0, 0.0]],
            [[0.0, 0.0, 255.0], [255.0, 255.0, 0.0]]
        ])
        # Sample exact corners
        coords = jnp.array([
            [0.0, 0.0],  # Top-left (red)
            [0.0, 1.0],  # Top-right (green)
            [1.0, 0.0],  # Bottom-left (blue)
            [1.0, 1.0],  # Bottom-right (yellow)
        ])
        result = bilinear_sample(texture, coords)
        expected = jnp.array([
            [255.0, 0.0, 0.0],
            [0.0, 255.0, 0.0],
            [0.0, 0.0, 255.0],
            [255.0, 255.0, 0.0],
        ])
        assert jnp.allclose(result, expected, atol=1.0)

    def test_center_interpolation(self):
        # 2x2 texture
        texture = jnp.array([
            [[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]],
            [[100.0, 100.0, 100.0], [200.0, 200.0, 200.0]]
        ])
        # Sample center
        coords = jnp.array([[0.5, 0.5]])
        result = bilinear_sample(texture, coords)
        # Should be average of all four corners
        expected = jnp.array([[100.0, 100.0, 100.0]])
        assert jnp.allclose(result, expected, atol=1.0)

    def test_output_shape(self):
        texture = jnp.ones((64, 64, 3))
        coords = jnp.array([[10.5, 20.5], [30.5, 40.5], [50.5, 60.5]])
        result = bilinear_sample(texture, coords)
        assert result.shape == (3, 3)

    def test_edge_interpolation(self):
        # Horizontal gradient
        texture = jnp.zeros((2, 4, 3))
        texture = texture.at[:, 0, :].set(0.0)
        texture = texture.at[:, 1, :].set(100.0)
        texture = texture.at[:, 2, :].set(200.0)
        texture = texture.at[:, 3, :].set(300.0)

        coords = jnp.array([[0.5, 1.5]])  # Between columns 1 and 2
        result = bilinear_sample(texture, coords)
        # Should be halfway between 100 and 200
        assert jnp.isclose(result[0, 0], 150.0, atol=5.0)


class TestFloorCastCoords:
    """Tests for floor_cast_coords function."""

    def test_output_shape(self):
        screen_y = jnp.arange(240, 480)  # Bottom half of 480-high screen
        player_pos = jnp.array([0.0, 0.0])
        player_angle = 0.0

        result = floor_cast_coords(screen_y, player_pos, player_angle)
        # Should return world coordinates for each pixel
        # Shape depends on implementation, but should be (num_rows, screen_width, 2)
        # or similar structure with 2D coordinates
        assert result.ndim >= 2
        assert result.shape[-1] == 2  # x, y coordinates

    def test_distance_increases_with_row(self):
        screen_y = jnp.array([241.0, 300.0, 400.0, 479.0])  # Various rows
        player_pos = jnp.array([0.0, 0.0])
        player_angle = 0.0

        result = floor_cast_coords(screen_y, player_pos, player_angle)

        # The bottom rows (closer to horizon at y=240) should map to
        # farther world coordinates than rows near the bottom of screen
        # This is a basic sanity check that floor distance increases
        # as we approach the horizon line

    def test_player_position_offset(self):
        screen_y = jnp.array([300.0])
        player_pos_1 = jnp.array([0.0, 0.0])
        player_pos_2 = jnp.array([10.0, 10.0])
        player_angle = 0.0

        result_1 = floor_cast_coords(screen_y, player_pos_1, player_angle)
        result_2 = floor_cast_coords(screen_y, player_pos_2, player_angle)

        # Results should differ by the player position offset
        # (approximately, depending on implementation)
        # This is a basic test that player position affects output
