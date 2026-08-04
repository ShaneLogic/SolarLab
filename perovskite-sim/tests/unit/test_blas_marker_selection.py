"""Marker parsing for the slow-suite BLAS thread limiter."""

import importlib.util
from pathlib import Path

import pytest


_CONFTEST_PATH = Path(__file__).resolve().parents[1] / "conftest.py"
_SPEC = importlib.util.spec_from_file_location("solarlab_root_conftest", _CONFTEST_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CONFTEST = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONFTEST)
_marker_selected_positively = _CONFTEST._marker_selected_positively


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("", False),
        ("not slow", False),
        ("not(slow)", False),
        ("fast and not slow", False),
        ("slow", True),
        ("slow and regression", True),
        ("slow or not slow", True),
        ("validation or not validation", True),
    ],
)
def test_marker_selected_positively(expression, expected):
    marker = "validation" if "validation" in expression else "slow"
    assert _marker_selected_positively(expression, marker) is expected
