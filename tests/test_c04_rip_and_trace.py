"""
Tests for Chapter 4: Rip and Trace — Ray Generation

Run with: uv run pytest tests/test_c04_rip_and_trace.py -v
"""

import math

import jax.numpy as jnp
import pytest

from solutions.c04_rip_and_trace import (
    angles_to_directions,
    rotate_vectors,
    normalize_vectors,
    scale_vectors,
)


class TestAnglesToDirections:
    """Tests for angles_to_directions function."""

    def test_cardinal_directions(self):
        angles = jnp.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
        result = angles_to_directions(angles)
        expected = jnp.array([
            [1.0, 0.0],   # East
            [0.0, 1.0],   # North
            [-1.0, 0.0],  # West
            [0.0, -1.0],  # South
        ])
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_diagonal_directions(self):
        angles = jnp.array([math.pi / 4, 3 * math.pi / 4])
        result = angles_to_directions(angles)
        sqrt2_2 = math.sqrt(2) / 2
        expected = jnp.array([
            [sqrt2_2, sqrt2_2],   # Northeast
            [-sqrt2_2, sqrt2_2],  # Northwest
        ])
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_output_shape(self):
        angles = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4])
        result = angles_to_directions(angles)
        assert result.shape == (5, 2)

    def test_unit_length(self):
        angles = jnp.linspace(0, 2 * math.pi, 10)
        result = angles_to_directions(angles)
        magnitudes = jnp.sqrt(jnp.sum(result ** 2, axis=1))
        assert jnp.allclose(magnitudes, 1.0, atol=1e-6)


class TestRotateVectors:
    """Tests for rotate_vectors function."""

    def test_90_degree_rotation(self):
        vecs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        result = rotate_vectors(vecs, math.pi / 2)
        expected = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_180_degree_rotation(self):
        vecs = jnp.array([[1.0, 0.0], [1.0, 1.0]])
        result = rotate_vectors(vecs, math.pi)
        expected = jnp.array([[-1.0, 0.0], [-1.0, -1.0]])
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_zero_rotation(self):
        vecs = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = rotate_vectors(vecs, 0.0)
        assert jnp.allclose(result, vecs, atol=1e-6)

    def test_preserves_magnitude(self):
        vecs = jnp.array([[3.0, 4.0], [5.0, 12.0]])
        original_mags = jnp.sqrt(jnp.sum(vecs ** 2, axis=1))
        result = rotate_vectors(vecs, 1.23)
        rotated_mags = jnp.sqrt(jnp.sum(result ** 2, axis=1))
        assert jnp.allclose(original_mags, rotated_mags, atol=1e-6)


class TestNormalizeVectors:
    """Tests for normalize_vectors function."""

    def test_basic(self):
        v = jnp.array([[3.0, 4.0], [0.0, 5.0]])
        result = normalize_vectors(v)
        expected = jnp.array([[0.6, 0.8], [0.0, 1.0]])
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_already_normalized(self):
        v = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
        result = normalize_vectors(v)
        assert jnp.allclose(result, v, atol=1e-6)

    def test_output_unit_length(self):
        v = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = normalize_vectors(v)
        magnitudes = jnp.sqrt(jnp.sum(result ** 2, axis=1))
        assert jnp.allclose(magnitudes, 1.0, atol=1e-6)

    def test_preserves_direction(self):
        v = jnp.array([[3.0, 4.0]])
        result = normalize_vectors(v)
        # Check that result points in same direction (positive dot product)
        dot = jnp.sum(v * result)
        assert dot > 0


class TestScaleVectors:
    """Tests for scale_vectors function."""

    def test_basic(self):
        v = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        scales = jnp.array([2.0, 0.5])
        result = scale_vectors(v, scales)
        expected = jnp.array([[2.0, 4.0], [1.5, 2.0]])
        assert jnp.allclose(result, expected)

    def test_scale_by_one(self):
        v = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        scales = jnp.array([1.0, 1.0])
        result = scale_vectors(v, scales)
        assert jnp.allclose(result, v)

    def test_scale_by_zero(self):
        v = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        scales = jnp.array([0.0, 0.0])
        result = scale_vectors(v, scales)
        expected = jnp.zeros((2, 2))
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        v = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scales = jnp.array([2.0, 3.0])
        result = scale_vectors(v, scales)
        assert result.shape == (2, 3)
