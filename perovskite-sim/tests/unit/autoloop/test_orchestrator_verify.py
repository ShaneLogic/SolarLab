# tests/unit/autoloop/test_orchestrator_verify.py
import dataclasses
from perovskite_sim.autoloop.types import Gap, Hypothesis, AblationMatrix, AblationProbe
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.orchestrator import attribute_top_gap


def _gap(gid="trend:Et_PVK ETL:V_oc"):
    return Gap(id=gid, metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _fake_ablation(gap, probe_runner):
    return AblationMatrix(gap_id=gap.id, baseline_val=40.0,
                          probes=(AblationProbe("f", "flag", {}, 40.0, 39.9, -0.1, True),))


class _LeadAttributor:
    """Returns an LLM lead (uncertain + a real cause)."""
    def attribute(self, gap, matrix, negatives):
        return Hypothesis(gap_id=gap.id, cause="physics",
                          mechanism="missing band-tail Urbach absorption", verdict="uncertain")


class _NoLeadAttributor:
    """Heuristic no-op uncertain (cause uncertain, no real mechanism)."""
    def attribute(self, gap, matrix, negatives):
        return Hypothesis(gap_id=gap.id, cause="uncertain",
                          mechanism="no single ablation lever identified", verdict="uncertain")


class _Verifier:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0
    def verify(self, hyp, gap, matrix):
        self.calls += 1
        return dataclasses.replace(hyp, verdict=self.verdict)


def _setup(tmp_path):
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap())
    led.save()


def test_lead_confirmed_sets_mechanism(tmp_path):
    _setup(tmp_path)
    v = _Verifier("confirmed")
    attribute_top_gap(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                      config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
                      cycle=1, timestamp="t", probe_runner_factory=lambda g: object(),
                      attributor=_LeadAttributor(), run_ablation_fn=_fake_ablation, verifier=v)
    assert v.calls == 1
    led = Ledger.load(tmp_path / "ledger")
    g = next(g for g in led.gaps if g.id == "trend:Et_PVK ETL:V_oc")
    assert g.mechanism == "missing band-tail Urbach absorption"   # confirmed -> mechanism set


def test_lead_refuted_adds_negative(tmp_path):
    _setup(tmp_path)
    attribute_top_gap(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                      config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
                      cycle=1, timestamp="t", probe_runner_factory=lambda g: object(),
                      attributor=_LeadAttributor(), run_ablation_fn=_fake_ablation,
                      verifier=_Verifier("refuted"))
    led = Ledger.load(tmp_path / "ledger")
    assert led.is_refuted("missing band-tail Urbach absorption")  # refuted -> negatives ledger


def test_heuristic_noop_skips_verifier(tmp_path):
    _setup(tmp_path)
    v = _Verifier("confirmed")
    attribute_top_gap(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                      config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
                      cycle=1, timestamp="t", probe_runner_factory=lambda g: object(),
                      attributor=_NoLeadAttributor(), run_ablation_fn=_fake_ablation, verifier=v)
    assert v.calls == 0                                            # cause=="uncertain" -> not verified


def test_no_verifier_is_unchanged(tmp_path):
    _setup(tmp_path)
    hyp = attribute_top_gap(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                            config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
                            cycle=1, timestamp="t", probe_runner_factory=lambda g: object(),
                            attributor=_LeadAttributor(), run_ablation_fn=_fake_ablation)
    assert hyp.verdict == "uncertain"                             # no verifier -> lead stays
