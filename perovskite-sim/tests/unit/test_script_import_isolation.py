from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "script",
    (
        "scripts/compare_scaps_defect_reference.py",
        "scripts/run_numerical_refinement.py",
        "scripts/verify_reproducibility.py",
    ),
)
def test_repository_cli_prefers_its_own_package_over_foreign_pythonpath(
    tmp_path: Path,
    script: str,
) -> None:
    foreign_package = tmp_path / "foreign" / "perovskite_sim"
    foreign_package.mkdir(parents=True)
    (foreign_package / "__init__.py").write_text(
        'raise RuntimeError("foreign checkout imported")\n',
        encoding="ascii",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(foreign_package.parent)

    completed = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "foreign checkout imported" not in completed.stderr
