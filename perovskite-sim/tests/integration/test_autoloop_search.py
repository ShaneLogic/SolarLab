# tests/integration/test_autoloop_search.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop.search import run_design_search, DesignKnob

REPO_ROOT = Path(__file__).resolve().parents[2]
REF = REPO_ROOT / "tests" / "integration" / "scaps_reference.json"
CFG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"


@pytest.mark.slow
def test_real_search_low_parity_gate(tmp_path):
    # Low parity-target so the gate passes (real parity ~0.69); tiny budget.
    cfg_before = CFG.read_text(encoding="utf-8")
    result = run_design_search(
        config_path=CFG, reference_path=REF, outputs_root=tmp_path / "out",
        timestamp="2026-06-16T00:00:00Z",
        space=[DesignKnob("etl_doping_cm3", 1e15, 1e19, "log")],
        budget=3, parity_target=0.5)
    assert result.n_evaluated == 3
    assert result.best is not None
    assert result.parity_overall >= 0.5             # gate passed
    assert CFG.read_text(encoding="utf-8") == cfg_before    # advisory — nothing applied
