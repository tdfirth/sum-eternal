"""
Unified Test Cases for Sum Eternal

Single source of truth for all test cases.
Used by both the game UI and pytest.
"""

import math
from typing import Any, Callable
import jax.numpy as jnp


# Test case format: (inputs_tuple, expected_output, test_name)
# - inputs_tuple: arguments to pass to function
# - expected_output: value to compare with jnp.allclose
# - test_name: short name for UI display

CHAPTERS = {
    1: {
        "module": "solutions.c01_first_blood",
        "functions": {
            "vector_sum": [
                (([1.0, 2.0, 3.0],), 6.0, "basic"),
                (([42.0],), 42.0, "single"),
                (([1.0, -2.0, 3.0, -4.0],), -2.0, "mixed"),
                (([0.0, 0.0, 0.0],), 0.0, "zeros"),
            ],
            "element_multiply": [
                (([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), [4.0, 10.0, 18.0], "basic"),
                (([1.0, 0.0, 3.0], [4.0, 5.0, 0.0]), [4.0, 0.0, 0.0], "zeros"),
                (([-1.0, 2.0, -3.0], [4.0, -5.0, -6.0]), [-4.0, -10.0, 18.0], "negative"),
            ],
            "dot_product": [
                (([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0, "basic"),
                (([1.0, 0.0], [0.0, 1.0]), 0.0, "orthogonal"),
                (([1.0, 2.0], [2.0, 4.0]), 10.0, "parallel"),
            ],
            "outer_product": [
                (([1.0, 2.0], [3.0, 4.0, 5.0]), [[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]], "basic"),
                (([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], "square"),
                (([1.0, 0.0], [2.0, 3.0]), [[2.0, 3.0], [0.0, 0.0]], "zeros"),
            ],
            "matrix_vector_mul": [
                (([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0]), [3.0, 7.0], "basic"),
                (([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [1.0, 2.0, 3.0]), [1.0, 2.0, 3.0], "identity"),
                (([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1.0, 0.0, 1.0]), [4.0, 10.0], "rectangular"),
            ],
            "matrix_matrix_mul": [
                (([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]), [[19.0, 22.0], [43.0, 50.0]], "basic"),
                (([[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 1.0]]), [[1.0, 2.0], [3.0, 4.0]], "identity"),
                (([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), [[22.0, 28.0], [49.0, 64.0]], "rectangular"),
            ],
        },
    },
    2: {
        "module": "solutions.c02_knee_deep_in_the_indices",
        "functions": {
            "transpose": [
                (([[1.0, 2.0], [3.0, 4.0]],), [[1.0, 3.0], [2.0, 4.0]], "square"),
                (([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], "rectangular"),
            ],
            "trace": [
                (([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],), 15.0, "basic"),
                (([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],), 4.0, "identity"),
                (([[3.0, 1.0], [2.0, 4.0]],), 7.0, "2x2"),
            ],
            "diag_extract": [
                (([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],), [1.0, 5.0, 9.0], "basic"),
                (([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],), [1.0, 1.0, 1.0, 1.0], "identity"),
                (([[3.0, 1.0], [2.0, 4.0]],), [3.0, 4.0], "2x2"),
            ],
            "sum_rows": [
                (([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],), [6.0, 15.0], "basic"),
                (([[1.0, 2.0, 3.0, 4.0]],), [10.0], "single_row"),
            ],
            "sum_cols": [
                (([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],), [5.0, 7.0, 9.0], "basic"),
                (([[1.0], [2.0], [3.0], [4.0]],), [10.0], "single_col"),
            ],
            "frobenius_norm_sq": [
                (([[1.0, 2.0], [3.0, 4.0]],), 30.0, "basic"),
                (([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],), 3.0, "identity"),
                (([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],), 0.0, "zeros"),
            ],
        },
    },
    3: {
        "module": "solutions.c03_the_slaughter_batch",
        "functions": {
            "batch_vector_sum": [
                (([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],), [6.0, 15.0], "basic"),
                (([[1.0, 2.0, 3.0]],), [6.0], "single"),
            ],
            "batch_dot_pairwise": [
                (([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]), [17.0, 53.0], "basic"),
                (([[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]), [0.0, 0.0], "orthogonal"),
            ],
            "batch_magnitude_sq": [
                (([[3.0, 4.0], [5.0, 12.0]],), [25.0, 169.0], "basic"),
                (([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]],), [1.0, 1.0, 1.0], "unit"),
                (([[0.0, 0.0], [1.0, 0.0]],), [0.0, 1.0], "zero"),
            ],
            "all_pairs_dot": [
                (([[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [2.0, 3.0]]), [[1.0, 2.0], [1.0, 3.0]], "basic"),
                (([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]), [[1.0, 0.0], [0.0, 1.0]], "identity_like"),
            ],
            "batch_matrix_vector": [
                (([[0.0, -1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), [[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]], "rotate90"),
                (([[1.0, 0.0], [0.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]), [[1.0, 2.0], [3.0, 4.0]], "identity"),
            ],
            "batch_outer": [
                (([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), [[[5.0, 6.0, 7.0], [10.0, 12.0, 14.0]], [[24.0, 27.0, 30.0], [32.0, 36.0, 40.0]]], "basic"),
                (([[1.0, 2.0]], [[3.0, 4.0]]), [[[3.0, 4.0], [6.0, 8.0]]], "single"),
            ],
        },
    },
    4: {
        "module": "solutions.c04_rip_and_trace",
        "functions": {
            "angles_to_directions": [
                (([0.0, math.pi / 2, math.pi, 3 * math.pi / 2],), [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], "cardinal"),
                (([math.pi / 4, 3 * math.pi / 4],), [[0.7071067811865476, 0.7071067811865476], [-0.7071067811865476, 0.7071067811865476]], "diagonal"),
            ],
            "rotate_vectors": [
                (([[1.0, 0.0], [0.0, 1.0]], math.pi / 2), [[0.0, 1.0], [-1.0, 0.0]], "90deg"),
                (([[1.0, 0.0], [1.0, 1.0]], math.pi), [[-1.0, 0.0], [-1.0, -1.0]], "180deg"),
                (([[1.0, 2.0], [3.0, 4.0]], 0.0), [[1.0, 2.0], [3.0, 4.0]], "zero"),
            ],
            "normalize_vectors": [
                (([[3.0, 4.0], [0.0, 5.0]],), [[0.6, 0.8], [0.0, 1.0]], "basic"),
                (([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]],), [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], "already_normal"),
            ],
            "scale_vectors": [
                (([[1.0, 2.0], [3.0, 4.0]], [2.0, 0.5]), [[2.0, 4.0], [1.5, 2.0]], "basic"),
                (([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0]), [[1.0, 2.0], [3.0, 4.0]], "identity"),
                (([[1.0, 2.0], [3.0, 4.0]], [0.0, 0.0]), [[0.0, 0.0], [0.0, 0.0]], "zero"),
            ],
        },
    },
}


def _to_jax(val: Any) -> Any:
    """Convert Python lists to JAX arrays."""
    if isinstance(val, (list, tuple)) and len(val) > 0:
        if isinstance(val[0], (list, int, float)):
            return jnp.array(val)
    return val


def run_test(func: Callable, inputs: tuple, expected: Any, atol: float = 1e-5) -> tuple[bool, str]:
    """
    Run a single test case.

    Returns:
        (passed, detail) where:
        - passed: True if test passed, False if failed, None if NotImplementedError
        - detail: result string or error message
    """
    # Convert inputs to JAX arrays
    jax_inputs = tuple(_to_jax(inp) for inp in inputs)
    expected_jax = _to_jax(expected)

    try:
        result = func(*jax_inputs)

        # Check if result matches expected
        if jnp.allclose(result, expected_jax, atol=atol):
            return (True, "pass")
        else:
            return (False, f"got {result}")

    except NotImplementedError:
        return (None, "not implemented")
    except Exception as e:
        return (False, f"error: {type(e).__name__}")


def run_function_tests(chapter: int, func_name: str) -> list[tuple[str, bool | None, str]]:
    """
    Run all tests for a function.

    Returns:
        List of (test_name, passed, detail) tuples
    """
    import importlib
    import sys

    chapter_data = CHAPTERS.get(chapter)
    if not chapter_data:
        return []

    module_name = chapter_data["module"]
    test_cases = chapter_data["functions"].get(func_name, [])

    # Force fresh import
    if module_name in sys.modules:
        del sys.modules[module_name]

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return [(name, None, "module not found") for _, _, name in test_cases]

    func = getattr(module, func_name, None)
    if func is None:
        return [(name, None, "function not found") for _, _, name in test_cases]

    results = []
    for inputs, expected, name in test_cases:
        passed, detail = run_test(func, inputs, expected)
        results.append((name, passed, detail))

    return results


def get_function_status(chapter: int, func_name: str) -> tuple[int, int, bool | None]:
    """
    Get status summary for a function.

    Returns:
        (passed_count, total_count, is_implemented)
        - is_implemented: True if working, False if error, None if NotImplementedError
    """
    results = run_function_tests(chapter, func_name)
    if not results:
        return (0, 0, None)

    total = len(results)
    passed = sum(1 for _, p, _ in results if p is True)

    # Check implementation status
    first_status = results[0][1]
    if first_status is None:
        is_implemented = None
    elif passed == total:
        is_implemented = True
    else:
        is_implemented = False

    return (passed, total, is_implemented)


def get_chapter_status(chapter: int) -> dict[str, tuple[int, int, bool | None]]:
    """
    Get status for all functions in a chapter.

    Returns:
        Dict mapping func_name -> (passed, total, is_implemented)
    """
    chapter_data = CHAPTERS.get(chapter)
    if not chapter_data:
        return {}

    return {
        func_name: get_function_status(chapter, func_name)
        for func_name in chapter_data["functions"]
    }


def get_chapter_functions(chapter: int) -> list[str]:
    """Get ordered list of function names for a chapter."""
    chapter_data = CHAPTERS.get(chapter)
    if not chapter_data:
        return []
    return list(chapter_data["functions"].keys())
