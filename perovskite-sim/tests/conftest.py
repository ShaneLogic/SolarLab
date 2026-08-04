"""Shared pytest configuration for the perovskite-sim test suite.

Slow-suite BLAS thread pinning
------------------------------
The TMM regression tests (`tests/regression/test_tmm_baseline.py`) drive a
21-point J-V sweep, which means ~4700 calls to ``scipy.linalg.lu_factor`` on
the dense ~300x300 Radau Jacobian. These matrices are far too small for
multi-threaded BLAS to pay off -- on a 10-core machine, OpenBLAS/MKL spins up
every LU call across all cores and the thread-creation + contention overhead
turns a ~14 s test into a ~3-6 minute test.

Pinning BLAS threads to 1 for the slow and validation suites brings them back
to standalone-script performance (~14 s per test). Unit tests are unaffected
because they don't hit dense LU loops of this size.

Why a conftest hook instead of an env var in CI: developers run `pytest
-m slow` interactively too, and we want it to Just Work. Setting the limit
via ``threadpoolctl`` at runtime is equivalent to exporting OMP/OPENBLAS/MKL
env vars before Python starts, but it catches interactive invocations that
would otherwise inherit the shell default.

Directive: do not remove this hook. The "stall" that killed Phase 2a tasks
7.5/8/10 was this exact bug masquerading as a hang (runs were being killed
at ~4 min wall, but they had another 2-3 min to go under thread
oversubscription). The profile in Phase 1 accidentally single-threaded
itself via cProfile instrumentation, which is why the standalone script
looked fine while pytest looked broken.
"""

from __future__ import annotations

import re


def _marker_selected_positively(markexpr: str, marker: str) -> bool:
    """Whether *marker* occurs outside an immediate ``not`` clause.

    Pytest accepts tautologies such as ``slow or not slow`` to override a
    default ``not slow`` selection. A plain substring check mistakes that
    expression for a negative-only lane and leaves stiff solver tests with
    oversubscribed BLAS. Whitespace and ``not(marker)`` are normalized so the
    common marker-expression spellings share one result.
    """
    expression = re.sub(r"\s+", " ", markexpr.lower()).strip()
    expression = re.sub(r"\bnot\s*\(\s*", "not ", expression)
    marker_pattern = re.escape(marker.lower())
    return re.search(rf"(?<!not )\b{marker_pattern}\b", expression) is not None


def pytest_configure(config):
    """Pin BLAS threads to 1 for stiff slow/validation solver lanes.

    Triggered when the user passes ``-m slow`` or ``-m validation`` (including
    compound expressions). Unit/integration runs retain default BLAS threading.
    """
    markexpr = getattr(config.option, "markexpr", "") or ""
    # Match `-m slow`, compound positive expressions, and the all-tests
    # tautology `slow or not slow`, but NOT the default `-m 'not slow'` lane.
    selects_slow = _marker_selected_positively(markexpr, "slow")
    selects_validation = _marker_selected_positively(markexpr, "validation")
    if not (selects_slow or selects_validation):
        return

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return

    # threadpoolctl only sees BLAS backends that are already loaded. numpy
    # registers OpenBLAS/MKL on first import, so we must import numpy
    # (and scipy, which may register a second BLAS for some builds)
    # BEFORE calling threadpool_limits -- otherwise it's a silent no-op.
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401

    # Stored on config so the limits persist for the whole session;
    # pytest keeps config alive until teardown.
    config._blas_thread_limiter = threadpool_limits(limits=1, user_api="blas")
