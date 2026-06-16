# tests/integration/test_autoloop_l3_seam.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop.orchestrator import guardian_once
from perovskite_sim.autoloop.reference import build_reference_source

REPO_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = REPO_ROOT / "tests" / "integration" / "scaps_lab_tiered.json"
CFG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"


@pytest.mark.slow
def test_guardian_scores_against_lab_base(tmp_path):
    report = guardian_once(
        ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
        reference_path=DESCRIPTOR, config_path=CFG, cycle=0,
        timestamp="2026-06-16T00:00:00Z", l0_paths=["tests/unit/autoloop"], baseline=None)
    assert report["overall"] is not None
    # the descriptor's lab base differs from the SCAPS base_model -> the loop
    # is anchoring absolutes to the (synthetic) lab device, end-to-end.
    lab_voc = build_reference_source(DESCRIPTOR).base_metrics()["Voc_V"]
    from perovskite_sim.autoloop.reference import ScapsReferenceSource
    scaps_voc = ScapsReferenceSource(REPO_ROOT / "tests" / "integration" / "scaps_reference.json").base_metrics()["Voc_V"]
    assert lab_voc != scaps_voc
