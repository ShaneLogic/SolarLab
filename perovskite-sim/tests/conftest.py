"""Shared pytest configuration for the perovskite-sim test suite.

Slow-suite BLAS thread pinning
------------------------------
The TMM regression tests (`tests/regression/test_tmm_baseline.py`) drive a
21-point J-V sweep, which means ~4700 calls to ``scipy.linalg.lu_factor`` on
the dense ~300x300 Radau Jacobian. These matrices are far too small for
multi-threaded BLAS to pay off -- on a 10-core machine, OpenBLAS/MKL spins up
every LU call across all cores and the thread-creation + contention overhead
turns a ~14 s test into a ~3-6 minute test.

Pinning BLAS threads to 1 for the slow suite brings it back to standalone-
script performance (~14 s per test). Unit tests are unaffected because they
don't hit dense LU loops of this size.

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


def pytest_configure(config):
    """Pin BLAS threads to 1 when the slow marker is selected.

    Triggered when the user passes `-m slow` (or any markexpr containing
    ``slow``). Unit/integration runs retain default BLAS threading.
    """
    markexpr = getattr(config.option, "markexpr", "") or ""
    # Match `-m slow`, `-m 'slow and X'`, `-m 'slow or Y'` but NOT
    # `-m 'not slow'` (the default unit-test run set via pyproject addopts).
    selects_slow = "slow" in markexpr and "not slow" not in markexpr
    if not selects_slow:
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
