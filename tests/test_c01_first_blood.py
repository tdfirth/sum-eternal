"""
Tests for Chapter 1: First Blood — Basic Contractions

These tests verify the fundamental einsum operations.
Run with: uv run pytest tests/test_c01_first_blood.py -v
"""

import jax.numpy as jnp
import pytest

from solutions.c01_first_blood import (
    vector_sum,
    element_multiply,
    dot_product,
    outer_product,
    matrix_vector_mul,
    matrix_matrix_mul,
)


class TestVectorSum:
    """Tests for vector_sum function."""

    def test_basic(self):
        v = jnp.array([1.0, 2.0, 3.0])
        result = vector_sum(v)
        assert jnp.isclose(result, 6.0)

    def test_single_element(self):
        v = jnp.array([42.0])
        result = vector_sum(v)
        assert jnp.isclose(result, 42.0)

    def test_negative_values(self):
        v = jnp.array([1.0, -2.0, 3.0, -4.0])
        result = vector_sum(v)
        assert jnp.isclose(result, -2.0)

    def test_zeros(self):
        v = jnp.array([0.0, 0.0, 0.0])
        result = vector_sum(v)
        assert jnp.isclose(result, 0.0)

    def test_output_is_scalar(self):
        v = jnp.array([1.0, 2.0, 3.0])
        result = vector_sum(v)
        assert result.shape == ()


class TestElementMultiply:
    """Tests for element_multiply function."""

    def test_basic(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        result = element_multiply(a, b)
        expected = jnp.array([4.0, 10.0, 18.0])
        assert jnp.allclose(result, expected)

    def test_with_zeros(self):
        a = jnp.array([1.0, 0.0, 3.0])
        b = jnp.array([4.0, 5.0, 0.0])
        result = element_multiply(a, b)
        expected = jnp.array([4.0, 0.0, 0.0])
        assert jnp.allclose(result, expected)

    def test_negative_values(self):
        a = jnp.array([-1.0, 2.0, -3.0])
        b = jnp.array([4.0, -5.0, -6.0])
        result = element_multiply(a, b)
        expected = jnp.array([-4.0, -10.0, 18.0])
        assert jnp.allclose(result, expected)

    def test_shape_preserved(self):
        a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = element_multiply(a, b)
        assert result.shape == (5,)


class TestDotProduct:
    """Tests for dot_product function."""

    def test_basic(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        result = dot_product(a, b)
        assert jnp.isclose(result, 32.0)  # 1*4 + 2*5 + 3*6

    def test_orthogonal_vectors(self):
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 1.0])
        result = dot_product(a, b)
        assert jnp.isclose(result, 0.0)

    def test_parallel_vectors(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([2.0, 4.0])  # 2 * a
        result = dot_product(a, b)
        expected = 2.0 + 8.0  # 1*2 + 2*4
        assert jnp.isclose(result, expected)

    def test_unit_vectors(self):
        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([1.0, 0.0, 0.0])
        result = dot_product(a, b)
        assert jnp.isclose(result, 1.0)

    def test_output_is_scalar(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0, 4.0])
        result = dot_product(a, b)
        assert result.shape == ()


class TestOuterProduct:
    """Tests for outer_product function."""

    def test_basic(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0, 4.0, 5.0])
        result = outer_product(a, b)
        expected = jnp.array([[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]])
        assert jnp.allclose(result, expected)

    def test_square_output(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([1.0, 2.0, 3.0])
        result = outer_product(a, b)
        expected = jnp.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0, 4.0, 5.0, 6.0])
        result = outer_product(a, b)
        assert result.shape == (2, 4)

    def test_with_zeros(self):
        a = jnp.array([1.0, 0.0])
        b = jnp.array([2.0, 3.0])
        result = outer_product(a, b)
        expected = jnp.array([[2.0, 3.0], [0.0, 0.0]])
        assert jnp.allclose(result, expected)


class TestMatrixVectorMul:
    """Tests for matrix_vector_mul function."""

    def test_basic(self):
        M = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        v = jnp.array([1.0, 1.0])
        result = matrix_vector_mul(M, v)
        expected = jnp.array([3.0, 7.0])
        assert jnp.allclose(result, expected)

    def test_identity_matrix(self):
        M = jnp.eye(3)
        v = jnp.array([1.0, 2.0, 3.0])
        result = matrix_vector_mul(M, v)
        assert jnp.allclose(result, v)

    def test_rectangular_matrix(self):
        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        v = jnp.array([1.0, 0.0, 1.0])
        result = matrix_vector_mul(M, v)
        expected = jnp.array([4.0, 10.0])  # [1+3, 4+6]
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        v = jnp.array([1.0, 1.0, 1.0])
        result = matrix_vector_mul(M, v)
        assert result.shape == (2,)


class TestMatrixMatrixMul:
    """Tests for matrix_matrix_mul function."""

    def test_basic(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = matrix_matrix_mul(A, B)
        expected = jnp.array([[19.0, 22.0], [43.0, 50.0]])
        assert jnp.allclose(result, expected)

    def test_identity(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        I = jnp.eye(2)
        result = matrix_matrix_mul(A, I)
        assert jnp.allclose(result, A)

    def test_rectangular(self):
        A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
        B = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
        result = matrix_matrix_mul(A, B)
        expected = jnp.array([[22.0, 28.0], [49.0, 64.0]])  # 2x2
        assert jnp.allclose(result, expected)

    def test_output_shape(self):
        A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
        B = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])  # 3x4
        result = matrix_matrix_mul(A, B)
        assert result.shape == (2, 4)
