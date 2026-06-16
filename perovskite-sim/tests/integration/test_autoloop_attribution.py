import pytest
from pathlib import Path
from perovskite_sim.autoloop.orchestrator import guardian_once, attribute_top_gap
from perovskite_sim.autoloop.attribution import HeuristicAttributor
from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner
from perovskite_sim.autoloop.ledger import Ledger

REPO_ROOT = Path(__file__).resolve().parents[2]
REF = REPO_ROOT / "tests" / "integration" / "scaps_reference.json"
CFG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"


@pytest.mark.slow
def test_attribution_pass_produces_hypothesis(tmp_path):
    # First a guardian cycle to populate the gap ledger.
    guardian_once(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                  reference_path=REF, config_path=CFG, cycle=0,
                  timestamp="2026-06-16T00:00:00Z", l0_paths=["tests/unit/autoloop"],
                  baseline=None)
    led = Ledger.load(tmp_path / "ledger")
    open_gaps = [g for g in led.gaps if g.status == "open"]
    if not open_gaps:
        pytest.skip("no open gaps produced on this config — nothing to attribute")

    top = max(open_gaps, key=lambda g: g.gap_mag)
    runner = SubprocessProbeRunner(config_path=CFG, reference_path=REF, gap=top)
    hyp = attribute_top_gap(
        ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
        config_path=CFG, reference_path=REF, cycle=1,
        timestamp="2026-06-16T00:00:00Z",
        probe_runner=runner, attributor=HeuristicAttributor())
    assert hyp is not None
    assert hyp.cause in {"bug", "numerics", "physics", "data", "uncertain"}
    assert (tmp_path / "out" / "attr-1" / "hypothesis.json").exists()
