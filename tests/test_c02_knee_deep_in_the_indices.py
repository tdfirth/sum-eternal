"""
Tests for Chapter 2: Knee Deep in the Indices — Reductions & Rearrangements

Run with: uv run pytest tests/test_c02_knee_deep_in_the_indices.py -v
"""

import jax.numpy as jnp
import pytest

from solutions.c02_knee_deep_in_the_indices import (
    transpose,
    trace,
    diag_extract,
    sum_rows,
    sum_cols,
    frobenius_norm_sq,
)


class TestTranspose:
    """Tests for transpose function."""

    def test_square_matrix(self):
        M = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = transpose(M)
        expected = jnp.array([[1.0, 3.0], [2.0, 4.0]])
        assert jnp.allclose(result, expected)

    def test_rectangular_matrix(self):
        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = transpose(M)
        expected = jnp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = transpose(M)
        assert result.shape == (3, 2)

    def test_double_transpose(self):
        M = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = transpose(transpose(M))
        assert jnp.allclose(result, M)


class TestTrace:
    """Tests for trace function."""

    def test_basic(self):
        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = trace(M)
        assert jnp.isclose(result, 15.0)  # 1 + 5 + 9

    def test_identity_matrix(self):
        M = jnp.eye(4)
        result = trace(M)
        assert jnp.isclose(result, 4.0)

    def test_2x2_matrix(self):
        M = jnp.array([[3.0, 1.0], [2.0, 4.0]])
        result = trace(M)
        assert jnp.isclose(result, 7.0)  # 3 + 4

    def test_output_is_scalar(self):
        M = jnp.eye(3)
        result = trace(M)
        assert result.shape == ()


class TestDiagExtract:
    """Tests for diag_extract function."""

    def test_basic(self):
        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = diag_extract(M)
        expected = jnp.array([1.0, 5.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_identity_matrix(self):
        M = jnp.eye(4)
        result = diag_extract(M)
        expected = jnp.array([1.0, 1.0, 1.0, 1.0])
        assert jnp.allclose(result, expected)

    def test_2x2_matrix(self):
        M = jnp.array([[3.0, 1.0], [2.0, 4.0]])
        result = diag_extract(M)
        expected = jnp.array([3.0, 4.0])
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        M = jnp.eye(5)
        result = diag_extract(M)
        assert result.shape == (5,)


class TestSumRows:
    """Tests for sum_rows function."""

    def test_basic(self):
        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = sum_rows(M)
        expected = jnp.array([6.0, 15.0])
        assert jnp.allclose(result, expected)

    def test_single_row(self):
        M = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        result = sum_rows(M)
        expected = jnp.array([10.0])
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = sum_rows(M)
        assert result.shape == (3,)


class TestSumCols:
    """Tests for sum_cols function."""

    def test_basic(self):
        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = sum_cols(M)
        expected = jnp.array([5.0, 7.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_single_column(self):
        M = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        result = sum_cols(M)
        expected = jnp.array([10.0])
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        M = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = sum_cols(M)
        assert result.shape == (4,)


class TestFrobeniusNormSq:
    """Tests for frobenius_norm_sq function."""

    def test_basic(self):
        M = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = frobenius_norm_sq(M)
        assert jnp.isclose(result, 30.0)  # 1 + 4 + 9 + 16

    def test_identity_matrix(self):
        M = jnp.eye(3)
        result = frobenius_norm_sq(M)
        assert jnp.isclose(result, 3.0)

    def test_zeros_matrix(self):
        M = jnp.zeros((3, 3))
        result = frobenius_norm_sq(M)
        assert jnp.isclose(result, 0.0)

    def test_rectangular_matrix(self):
        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = frobenius_norm_sq(M)
        assert jnp.isclose(result, 91.0)  # 1+4+9+16+25+36

    def test_output_is_scalar(self):
        M = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = frobenius_norm_sq(M)
        assert result.shape == ()
