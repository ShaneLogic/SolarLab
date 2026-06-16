from perovskite_sim.autoloop.types import Gap, Hypothesis, ImplementResult
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.orchestrator import run_boulder


def _gap(gid, mag):
    return Gap(id=gid, metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=mag, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def test_sweep_drains_all_gaps_and_records_proposals(tmp_path):
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap("g_hi", 0.5))
    led.add_gap(_gap("g_lo", 0.2))
    led.save()

    def sense(cycle):
        return 0.4

    def attribute(cycle):
        # write a confirmed hypothesis for the current top open gap
        led = Ledger.load(tmp_path / "ledger")
        top = max((g for g in led.gaps if g.status == "open"), key=lambda g: g.gap_mag)
        led.add_hypothesis(Hypothesis(gap_id=top.id, cause="physics",
                                      mechanism="flag X term", verdict="confirmed"))
        led.save()

    def implement(cycle, apply):
        led = Ledger.load(tmp_path / "ledger")
        top = max((g for g in led.gaps if g.status == "open"), key=lambda g: g.gap_mag)
        return ImplementResult("dry_run", top.id, "interface_plane_projection", (), None)

    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="2026-06-16T00:00:00Z", converge=False,
                         sense=sense, attribute=attribute, implement=implement)

    assert result.mode == "sweep"
    assert result.stop_reason == "sweep_complete"
    assert result.cycles == 2                       # both gaps drained
    assert {p.gap_id for p in result.proposals} == {"g_hi", "g_lo"}
    assert result.landed_count == 0                 # dry-run, nothing committed
    # both gaps marked attempted
    led2 = Ledger.load(tmp_path / "ledger")
    assert all(g.status == "attempted" for g in led2.gaps)


def test_sweep_empty_when_no_gaps(tmp_path):
    led = Ledger(root=tmp_path / "ledger"); led.save()
    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="2026-06-16T00:00:00Z", converge=False,
                         sense=lambda c: 0.95, attribute=lambda c: None,
                         implement=lambda c, apply: None)
    assert result.cycles == 0 and result.proposals == ()
