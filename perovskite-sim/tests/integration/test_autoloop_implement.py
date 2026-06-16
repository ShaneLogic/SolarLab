# tests/integration/test_autoloop_implement.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop.orchestrator import implement_top_confirmed
from perovskite_sim.autoloop.types import GateVerdict
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.types import Gap, Hypothesis

REPO_ROOT = Path(__file__).resolve().parents[2]
CFG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"


@pytest.mark.slow
def test_implement_dry_run_on_confirmed_gap(tmp_path):
    # Seed a confirmed hypothesis for a promotable lever.
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(Gap(id="trend:Nt_PVK ETL:V_oc", metric="V_oc", sweep="Nt_PVK ETL",
                    sweep_point=0.0, solarlab_val=30.0, reference_val=70.0, gap_mag=0.4,
                    kind="trend", status="open", found_cycle=0, last_attempt_cycle=0,
                    mechanism=None))
    led.add_hypothesis(Hypothesis(gap_id="trend:Nt_PVK ETL:V_oc", cause="physics",
                                  mechanism="flag SOLARLAB_IFACE_PROJ term", verdict="confirmed"))
    led.save()

    # Reduced real gate runner: G1 only (fast pytest subset) — skips the expensive
    # full-suite G0 (validated separately). Confirms the real propose+edit+revert path.
    from perovskite_sim.autoloop.ladder import run_l0

    def reduced_gates(edit, gap, hyp):
        ok, detail = run_l0(["tests/unit/autoloop"])
        return [GateVerdict("G1_numerics", ok, detail)]

    cfg_text_before = CFG.read_text(encoding="utf-8")
    r = implement_top_confirmed(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                                config_path=CFG, reference_path=tmp_path / "r.json",
                                cycle=1, timestamp="2026-06-16T00:00:00Z",
                                gate_runner=reduced_gates, apply=False)
    assert r.status in {"dry_run", "gates_failed"}
    assert r.device_key == "interface_plane_projection"
    assert CFG.read_text(encoding="utf-8") == cfg_text_before    # parity config untouched after dry-run
