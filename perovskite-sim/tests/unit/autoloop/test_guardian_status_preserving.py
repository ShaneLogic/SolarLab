# tests/unit/autoloop/test_guardian_status_preserving.py
from perovskite_sim.autoloop.types import Gap, ParityScore, SweepScore, LadderResult
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.orchestrator import guardian_once


def _gap(gid, status):
    return Gap(id=gid, metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status=status, found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _ladder_with_nd_etl_gap(**kw):
    # A ladder result whose scorecard yields a trend:Nd_ETL:V_oc gap (low closure).
    score = ParityScore(overall=0.4, base_deltas={},
                        per_sweep={"Nd_ETL": SweepScore("Nd_ETL", 30.0, 5, 4)})
    return LadderResult(l0_pass=True, l1_pass=True, score=score, details={})


def test_resense_does_not_resurrect_attempted_gap(tmp_path):
    # Seed the ledger with the gap already marked attempted.
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap("trend:Nd_ETL:V_oc", "attempted"))
    led.save()

    guardian_once(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                  reference_path=tmp_path / "r.json", config_path=tmp_path / "c.yaml",
                  cycle=1, timestamp="2026-06-16T00:00:00Z", l0_paths=["tests/unit/autoloop"],
                  baseline=None, run_ladder_fn=_ladder_with_nd_etl_gap)

    led2 = Ledger.load(tmp_path / "ledger")
    g = next(g for g in led2.gaps if g.id == "trend:Nd_ETL:V_oc")
    assert g.status == "attempted"     # preserved, NOT reset to open


def test_new_gap_enters_open(tmp_path):
    led = Ledger(root=tmp_path / "ledger"); led.save()   # empty
    guardian_once(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                  reference_path=tmp_path / "r.json", config_path=tmp_path / "c.yaml",
                  cycle=0, timestamp="2026-06-16T00:00:00Z", l0_paths=["tests/unit/autoloop"],
                  baseline=None, run_ladder_fn=_ladder_with_nd_etl_gap)
    led2 = Ledger.load(tmp_path / "ledger")
    g = next(g for g in led2.gaps if g.id == "trend:Nd_ETL:V_oc")
    assert g.status == "open"          # brand-new gap enters open
