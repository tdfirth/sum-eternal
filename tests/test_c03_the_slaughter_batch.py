"""
Tests for Chapter 3: The Slaughter Batch — Batch Operations

Run with: uv run pytest tests/test_c03_the_slaughter_batch.py -v
"""

import jax.numpy as jnp
import pytest

from solutions.c03_the_slaughter_batch import (
    batch_vector_sum,
    batch_dot_pairwise,
    batch_magnitude_sq,
    all_pairs_dot,
    batch_matrix_vector,
    batch_outer,
)


class TestBatchVectorSum:
    """Tests for batch_vector_sum function."""

    def test_basic(self):
        batch = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = batch_vector_sum(batch)
        expected = jnp.array([6.0, 15.0])
        assert jnp.allclose(result, expected)

    def test_single_batch(self):
        batch = jnp.array([[1.0, 2.0, 3.0]])
        result = batch_vector_sum(batch)
        expected = jnp.array([6.0])
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        result = batch_vector_sum(batch)
        assert result.shape == (4,)


class TestBatchDotPairwise:
    """Tests for batch_dot_pairwise function."""

    def test_basic(self):
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = batch_dot_pairwise(a, b)
        expected = jnp.array([17.0, 53.0])  # [1*5+2*6, 3*7+4*8]
        assert jnp.allclose(result, expected)

    def test_orthogonal_vectors(self):
        a = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        result = batch_dot_pairwise(a, b)
        expected = jnp.array([0.0, 0.0])
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        b = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        result = batch_dot_pairwise(a, b)
        assert result.shape == (3,)


class TestBatchMagnitudeSq:
    """Tests for batch_magnitude_sq function."""

    def test_basic(self):
        v = jnp.array([[3.0, 4.0], [5.0, 12.0]])
        result = batch_magnitude_sq(v)
        expected = jnp.array([25.0, 169.0])
        assert jnp.allclose(result, expected)

    def test_unit_vectors(self):
        v = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
        result = batch_magnitude_sq(v)
        expected = jnp.array([1.0, 1.0, 1.0])
        assert jnp.allclose(result, expected)

    def test_zero_vector(self):
        v = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        result = batch_magnitude_sq(v)
        expected = jnp.array([0.0, 1.0])
        assert jnp.allclose(result, expected)


class TestAllPairsDot:
    """Tests for all_pairs_dot function."""

    def test_basic(self):
        a = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b = jnp.array([[1.0, 1.0], [2.0, 3.0]])
        result = all_pairs_dot(a, b)
        expected = jnp.array([[1.0, 2.0], [1.0, 3.0]])
        assert jnp.allclose(result, expected)

    def test_rectangular_output(self):
        a = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3 vectors
        b = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # 2 vectors
        result = all_pairs_dot(a, b)
        assert result.shape == (3, 2)

    def test_same_vectors(self):
        a = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        result = all_pairs_dot(a, a)
        expected = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # identity-like
        assert jnp.allclose(result, expected)


class TestBatchMatrixVector:
    """Tests for batch_matrix_vector function."""

    def test_basic(self):
        M = jnp.array([[0.0, -1.0], [1.0, 0.0]])  # 90-degree rotation
        batch = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        result = batch_matrix_vector(M, batch)
        expected = jnp.array([[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]])
        assert jnp.allclose(result, expected)

    def test_identity_matrix(self):
        M = jnp.eye(2)
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = batch_matrix_vector(M, batch)
        assert jnp.allclose(result, batch)

    def test_output_shape(self):
        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        batch = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = batch_matrix_vector(M, batch)
        assert result.shape == (2, 3)


class TestBatchOuter:
    """Tests for batch_outer function."""

    def test_basic(self):
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
        result = batch_outer(a, b)
        expected = jnp.array([
            [[5.0, 6.0, 7.0], [10.0, 12.0, 14.0]],
            [[24.0, 27.0, 30.0], [32.0, 36.0, 40.0]]
        ])
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        a = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 x 2
        b = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])  # 3 x 4
        result = batch_outer(a, b)
        assert result.shape == (3, 2, 4)

    def test_single_batch(self):
        a = jnp.array([[1.0, 2.0]])
        b = jnp.array([[3.0, 4.0]])
        result = batch_outer(a, b)
        expected = jnp.array([[[3.0, 4.0], [6.0, 8.0]]])
        assert jnp.allclose(result, expected)
