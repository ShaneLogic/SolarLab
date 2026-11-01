"""Test import isolation and job cleanup through fresh pytest processes."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


PROJECT = Path(__file__).resolve().parents[2]


def _run_case(tmp_path: Path, body: str, *, backend_scope: str | None) -> None:
    conftest = (PROJECT / "tests/conftest.py").read_text()
    if backend_scope is not None:
        conftest += (
            "\n"
            + (PROJECT / "tests" / backend_scope / "backend/conftest.py").read_text()
        )
    (tmp_path / "conftest.py").write_text(conftest)
    (tmp_path / "test_case.py").write_text(textwrap.dedent(body))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(PROJECT)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_pytest_does_not_import_backend(tmp_path: Path) -> None:
    _run_case(
        tmp_path,
        """
        import sys
        import numpy
        import scipy
        import perovskite_sim

        def test_core():
            assert not any(
                name == "backend" or name.startswith("backend.")
                for name in sys.modules
            )
        """,
        backend_scope=None,
    )


@pytest.mark.parametrize("backend_scope", ["unit", "integration"])
def test_backend_cleanup_precedes_monkeypatch_undo(
    tmp_path: Path, backend_scope: str
) -> None:
    _run_case(
        tmp_path,
        """
        import time
        from types import SimpleNamespace
        from backend.jobs import JobRegistry

        state = SimpleNamespace(value="original", observed=[])

        def test_submit(monkeypatch):
            monkeypatch.setattr(state, "value", "patched")
            def job(_reporter):
                time.sleep(0.02)
                state.observed.append(state.value)
                return {}
            JobRegistry().submit(job)

        def test_finished():
            assert state.observed == ["patched"]
            assert state.value == "original"
        """,
        backend_scope=backend_scope,
    )
