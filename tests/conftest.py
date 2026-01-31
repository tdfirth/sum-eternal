"""
Pytest configuration for Sum Eternal tests.
"""

import sys
from pathlib import Path

import pytest


# Ensure solutions directory is importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def rtol():
    """Default relative tolerance for floating point comparisons."""
    return 1e-5


@pytest.fixture
def atol():
    """Default absolute tolerance for floating point comparisons."""
    return 1e-8
