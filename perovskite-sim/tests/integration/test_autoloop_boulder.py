# tests/integration/test_autoloop_boulder.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop.orchestrator import (
    run_boulder, guardian_once, attribute_top_gap, implement_top_confirmed)
from perovskite_sim.autoloop.attribution import HeuristicAttributor
from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner
from perovskite_sim.autoloop.gates_impl import make_implement_gate_runner

REPO_ROOT = Path(__file__).resolve().parents[2]
REF = REPO_ROOT / "tests" / "integration" / "scaps_reference.json"
CFG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"


@pytest.mark.slow
def test_boulder_sweep_real(tmp_path):
    lr, orr = tmp_path / "ledger", tmp_path / "out"

    def sense(cycle):
        rep = guardian_once(ledger_root=lr, outputs_root=orr, reference_path=REF,
                            config_path=CFG, cycle=cycle, timestamp="t",
                            l0_paths=["tests/unit/autoloop"], baseline=None)
        return rep["overall"]

    def attribute(cycle):
        attribute_top_gap(ledger_root=lr, outputs_root=orr, config_path=CFG,
                          reference_path=REF, cycle=cycle, timestamp="t",
                          probe_runner_factory=lambda g: SubprocessProbeRunner(
                              config_path=CFG, reference_path=REF, gap=g),
                          attributor=HeuristicAttributor())

    def implement(cycle, apply):
        def _measure(edit, gap):
            return SubprocessProbeRunner(config_path=edit.config_path, reference_path=REF,
                                         gap=gap).run({"env_flags": {}, "jv_overrides": {}, "measure": "gap"})
        return implement_top_confirmed(
            ledger_root=lr, outputs_root=orr, config_path=CFG, reference_path=REF,
            cycle=cycle, timestamp="t",
            gate_runner=make_implement_gate_runner(measure_badness=_measure), apply=apply)

    cfg_before = CFG.read_text(encoding="utf-8")
    result = run_boulder(ledger_root=lr, outputs_root=orr, timestamp="t",
                         converge=False, sense=sense, attribute=attribute, implement=implement)
    assert result.mode == "sweep" and result.stop_reason == "sweep_complete"
    # The sweep drains every open gap into exactly one proposal and marks it
    # "attempted" in the ledger. WHICH gaps are open evolves as parity
    # improves — the originally pinned trend:Nd_ETL:V_oc closed after the
    # SS interface-channel calibration — so assert the structural contract
    # (proposals == attempted ledger gaps) rather than a specific gap id.
    from perovskite_sim.autoloop.ledger import Ledger
    led = Ledger.load(lr)
    proposal_ids = [p.gap_id for p in result.proposals]
    attempted_ids = {g.id for g in led.gaps if g.status == "attempted"}
    assert set(proposal_ids) == attempted_ids
    assert len(proposal_ids) == len(set(proposal_ids))  # one proposal per gap
    assert CFG.read_text(encoding="utf-8") == cfg_before     # sweep commits nothing
