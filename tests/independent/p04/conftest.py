"""R3 independent P04 oracles: single-thread BLAS and import path.

The tests here are fast unit oracles (no solves, no sweeps), but the brief
requires single-threaded numerics, so BLAS is pinned for the whole directory.
threadpoolctl only sees backends that are already loaded, hence numpy and
scipy.linalg are imported first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy  # noqa: F401  (must precede threadpool_limits)
import scipy.linalg  # noqa: F401
import pytest
from threadpoolctl import threadpool_limits

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


@pytest.fixture(scope="session", autouse=True)
def _single_thread_blas():
    with threadpool_limits(limits=1):
        yield
