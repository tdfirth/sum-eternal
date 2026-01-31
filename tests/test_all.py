"""
Unified pytest wrapper for Sum Eternal.

Uses test cases from solutions/test_cases.py.
Run with: uv run pytest tests/test_all.py -v
"""

import pytest
from solutions.test_cases import CHAPTERS, run_function_tests


def generate_test_cases():
    """Generate pytest parameters from unified test cases."""
    for chapter_num, chapter_data in CHAPTERS.items():
        module = chapter_data["module"]
        for func_name in chapter_data["functions"]:
            yield pytest.param(
                chapter_num,
                func_name,
                id=f"ch{chapter_num}_{func_name}"
            )


@pytest.mark.parametrize("chapter,func_name", list(generate_test_cases()))
def test_function(chapter: int, func_name: str):
    """Test a single function using unified test cases."""
    results = run_function_tests(chapter, func_name)

    # Check if any tests failed
    failures = []
    not_implemented = False

    for test_name, passed, detail in results:
        if passed is None:
            not_implemented = True
            break
        elif passed is False:
            failures.append(f"{test_name}: {detail}")

    if not_implemented:
        pytest.skip("Not implemented")

    if failures:
        pytest.fail("\n".join(failures))
