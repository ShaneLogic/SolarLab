# tests/unit/autoloop/test_orchestrator.py
from perovskite_sim.autoloop.types import LadderResult, ParityScore, SweepScore
from perovskite_sim.autoloop.orchestrator import guardian_once


def _score(overall=0.40):
    return ParityScore(overall=overall, base_deltas={"V_oc": -0.07},
                       per_sweep={"Nd_ETL": SweepScore("Nd_ETL", 30.0, 5, 4)})


def test_guardian_once_records_gaps_and_returns_verdicts(tmp_path):
    fake_ladder = LadderResult(l0_pass=True, l1_pass=True, score=_score(), details={})
    report = guardian_once(
        ledger_root=tmp_path / "ledger",
        outputs_root=tmp_path / "out",
        reference_path=tmp_path / "ref.json",
        config_path=tmp_path / "c.yaml",
        cycle=0,
        timestamp="2026-06-16T00:00:00Z",
        run_ladder_fn=lambda **kw: fake_ladder,
        baseline=_score(0.80),       # current 0.40 << baseline 0.80 -> G3 fails
    )
    assert report["gate_passed"] is False           # parity regressed
    assert any("Nd_ETL" in g for g in report["gap_ids"])
    assert (tmp_path / "ledger" / "gaps.json").exists()
    assert (tmp_path / "out" / "run-0" / "report.json").exists()


def test_guardian_once_seeds_negative_results(tmp_path):
    fake_ladder = LadderResult(l0_pass=True, l1_pass=True, score=_score(0.9), details={})
    guardian_once(
        ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
        reference_path=tmp_path / "ref.json", config_path=tmp_path / "c.yaml",
        cycle=0, timestamp="2026-06-16T00:00:00Z",
        run_ladder_fn=lambda **kw: fake_ladder, baseline=None,
    )
    from perovskite_sim.autoloop.ledger import Ledger
    led = Ledger.load(tmp_path / "ledger")
    assert led.is_refuted("DOS-cap projection target") is True
