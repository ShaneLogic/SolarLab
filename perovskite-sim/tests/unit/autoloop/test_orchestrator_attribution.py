# tests/unit/autoloop/test_orchestrator_attribution.py
from perovskite_sim.autoloop.types import Gap, AblationMatrix, AblationProbe, Hypothesis
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.orchestrator import attribute_top_gap


def _gap(gid, mag, status="open"):
    return Gap(id=gid, metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=mag, kind="trend",
               status=status, found_cycle=0, last_attempt_cycle=0, mechanism=None)


class _FakeAttributor:
    def attribute(self, gap, matrix, negatives):
        return Hypothesis(gap_id=gap.id, cause="physics", mechanism="flag X term",
                          verdict="confirmed", verifier_votes=1)


def _fake_ablation(gap, probe_runner):
    return AblationMatrix(gap_id=gap.id, baseline_val=40.0,
                          probes=(AblationProbe("f", "flag", {}, 40.0, 22.0, -18.0, True),))


def test_attribute_top_gap_picks_highest_open_and_records(tmp_path):
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap("trend:Nd_ETL:V_oc", 0.4))
    led.add_gap(_gap("absolute:base:V_oc", 0.1))
    led.save()

    hyp = attribute_top_gap(
        ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
        config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
        cycle=1, timestamp="2026-06-16T00:00:00Z",
        probe_runner=object(), attributor=_FakeAttributor(),
        run_ablation_fn=_fake_ablation,
    )
    assert hyp is not None and hyp.gap_id == "trend:Nd_ETL:V_oc"   # highest gap_mag

    led2 = Ledger.load(tmp_path / "ledger")
    assert any(h.gap_id == "trend:Nd_ETL:V_oc" for h in led2.hypotheses)
    g = next(g for g in led2.gaps if g.id == "trend:Nd_ETL:V_oc")
    assert g.mechanism == "flag X term"                            # confirmed -> mechanism written
    assert (tmp_path / "out" / "attr-1" / "hypothesis.json").exists()


def test_attribute_top_gap_skips_non_open_and_returns_none(tmp_path):
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap("g1", 0.9, status="blocked"))
    led.add_gap(_gap("g2", 0.8, status="refuted"))
    led.save()

    hyp = attribute_top_gap(
        ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
        config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
        cycle=1, timestamp="2026-06-16T00:00:00Z",
        probe_runner=object(), attributor=_FakeAttributor(),
        run_ablation_fn=_fake_ablation,
    )
    assert hyp is None
