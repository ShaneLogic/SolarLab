"""One-command test environment and explicit optional prerequisites."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import shutil

import pytest

from suite_support import changed_bound_core_sources, clean_core_skip_reason


ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"


@pytest.fixture(scope="session", autouse=True)
def single_thread_blas():
    # Collection/plugins may have imported BLAS before conftest was loaded.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield


@lru_cache(maxsize=1)
def _clean_core_reason() -> str | None:
    try:
        return clean_core_skip_reason(changed_bound_core_sources(ROOT))
    except RuntimeError as exc:
        return str(exc)


def pytest_collection_modifyitems(items):
    # Mark before any module-scoped catalog/compiler fixture can run.
    core_reason = _clean_core_reason() if any(
        item.get_closest_marker("requires_clean_core") for item in items
    ) else None
    required = tuple(ROOT / "web/node_modules" / name / "package.json"
                     for name in ("typescript", "jsdom", "json-schema-to-typescript"))
    missing_frontend = not shutil.which("node") or any(not path.is_file() for path in required)
    for item in items:
        if core_reason and item.get_closest_marker("requires_clean_core"):
            item.add_marker(pytest.mark.skip(reason=core_reason))
        if missing_frontend and item.get_closest_marker("frontend"):
            reason = "frontend contracts require Node and web npm ci dependencies (typescript, jsdom, json-schema-to-typescript)"
            if os.environ.get("PYTEST_CI_JOB") == "frontend":
                raise pytest.UsageError(reason)
            item.add_marker(pytest.mark.skip(reason=reason))
