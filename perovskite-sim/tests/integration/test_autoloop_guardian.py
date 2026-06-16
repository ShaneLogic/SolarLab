# tests/integration/test_autoloop_guardian.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop import guardian_once

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_guardian_once_real_solver_produces_report(tmp_path):
    """Full sense-and-record cycle on the real scaps_mirror_v2 config.

    Asserts the ladder ran the real solver, produced a parity score, and
    wrote a report — without asserting a specific parity number (that is
    the moving target the loop tracks).
    """
    report = guardian_once(
        ledger_root=tmp_path / "ledger",
        outputs_root=tmp_path / "out",
        reference_path=REPO_ROOT / "tests" / "integration" / "scaps_reference.json",
        config_path=REPO_ROOT / "configs" / "scaps_mirror_v2.yaml",
        cycle=0,
        timestamp="2026-06-16T00:00:00Z",
        l0_paths=["tests/unit/autoloop"],     # keep L0 fast inside the smoke
        baseline=None,
    )
    assert report["overall"] is not None
    assert 0.0 <= report["overall"] <= 1.0
    assert (tmp_path / "out" / "run-0" / "report.json").exists()
    assert (tmp_path / "ledger" / "gaps.json").exists()
