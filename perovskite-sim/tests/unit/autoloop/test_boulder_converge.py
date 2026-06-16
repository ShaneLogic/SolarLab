# tests/unit/autoloop/test_boulder_converge.py
from perovskite_sim.autoloop.types import Gap, Hypothesis, ImplementResult
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.orchestrator import run_boulder


def _seed(tmp_path, *gaps):
    led = Ledger(root=tmp_path / "ledger")
    for gid, mag in gaps:
        led.add_gap(Gap(id=gid, metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
                        solarlab_val=30.0, reference_val=70.0, gap_mag=mag, kind="trend",
                        status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None))
    led.save()


def _confirm_top(tmp_path):
    led = Ledger.load(tmp_path / "ledger")
    og = [g for g in led.gaps if g.status == "open"]
    if og:
        top = max(og, key=lambda g: g.gap_mag)
        led.add_hypothesis(Hypothesis(gap_id=top.id, cause="physics",
                                      mechanism="flag X term", verdict="confirmed"))
        led.save()


def test_converge_stops_success_when_parity_met(tmp_path):
    _seed(tmp_path, ("g", 0.5))
    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="t", converge=True, parity_target=0.9,
                         sense=lambda c: 0.95, attribute=lambda c: None,
                         implement=lambda c, a: None)
    assert result.stop_reason == "success" and result.cycles == 0


def test_converge_stops_drained_when_no_open_gaps(tmp_path):
    _seed(tmp_path)   # no gaps
    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="t", converge=True, parity_target=0.9,
                         sense=lambda c: 0.5, attribute=lambda c: None,
                         implement=lambda c, a: None)
    assert result.stop_reason == "drained"


def test_converge_lands_then_drains(tmp_path):
    _seed(tmp_path, ("g", 0.5))

    def implement(cycle, apply):
        # land the fix: close the gap (as implement_top_confirmed would on apply)
        led = Ledger.load(tmp_path / "ledger")
        g = next(x for x in led.gaps if x.id == "g")
        led.add_gap(g.with_status("closed"))
        led.save()
        return ImplementResult("applied", "g", "interface_plane_projection", (), "sha123")

    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="t", converge=True, parity_target=0.99, max_cycles=5,
                         sense=lambda c: 0.5, attribute=lambda c: _confirm_top(tmp_path),
                         implement=implement)
    assert result.landed_count == 1
    assert result.stop_reason == "drained"        # after landing, no open gaps left


def test_converge_stops_cap_when_never_improves(tmp_path):
    _seed(tmp_path, ("g", 0.5))

    def implement(cycle, apply):
        return ImplementResult("not_promotable", "g", None, (), None)

    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="t", converge=True, parity_target=0.9, max_cycles=3,
                         # re-seed an open gap each sense so it never drains
                         sense=lambda c: (_seed(tmp_path, ("g2", 0.4)) or 0.5),
                         attribute=lambda c: None, implement=implement)
    assert result.stop_reason == "cap" and result.cycles == 3


def test_converge_stops_halt_on_reject_streak(tmp_path):
    _seed(tmp_path, ("g", 0.5))

    def implement(cycle, apply):
        return ImplementResult("gates_failed", "g", "interface_plane_projection", (), None)

    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="t", converge=True, parity_target=0.9, max_cycles=10,
                         reject_streak=2,
                         sense=lambda c: (_seed(tmp_path, (f"g{c}", 0.4)) or 0.5),
                         attribute=lambda c: _confirm_top(tmp_path), implement=implement)
    assert result.stop_reason == "halt"
