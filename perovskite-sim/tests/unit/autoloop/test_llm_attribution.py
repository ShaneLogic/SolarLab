# tests/unit/autoloop/test_llm_attribution.py
from perovskite_sim.autoloop.types import Gap, AblationProbe, AblationMatrix, Hypothesis, NegativeResult
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.cognition import FakeRuntime
from perovskite_sim.autoloop.llm_attribution import (
    LLMAttributor, build_attribution_prompt, ATTRIBUTION_SCHEMA,
)


def _gap():
    return Gap(id="trend:Et_PVK ETL:V_oc", metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _uncertain_matrix():
    # all probes flat -> heuristic returns "uncertain"
    probes = (
        AblationProbe("SOLARLAB_IFACE_PROJ", "flag", {}, 40.0, 39.9, -0.1, True),
        AblationProbe("grid_n80", "grid", {}, 40.0, 40.1, 0.1, True),
        AblationProbe("dark_jsc", "limiting", {}, 0.0, 0.01, 0.01, True),
    )
    return AblationMatrix(gap_id="trend:Et_PVK ETL:V_oc", baseline_val=40.0, probes=probes)


class _SpyRuntime:
    def __init__(self, out):
        self.out = out
        self.calls = 0
    def complete(self, prompt, schema):
        self.calls += 1
        self.last_prompt = prompt
        return dict(self.out)


def test_heuristic_confirmed_skips_llm(tmp_path):
    # a flag probe that strongly improves -> heuristic confirms physics -> LLM NOT called
    from perovskite_sim.autoloop.types import AblationMatrix as AM, AblationProbe as AP
    matrix = AM(gap_id="g", baseline_val=40.0, probes=(
        AP("SOLARLAB_IFACE_PROJ", "flag", {}, 40.0, 20.0, -20.0, True),
        AP("grid_n80", "grid", {}, 40.0, 40.0, 0.0, True),
        AP("dark_jsc", "limiting", {}, 0.0, 0.0, 0.0, True)))
    spy = _SpyRuntime({"cause": "physics", "mechanism": "x"})
    hyp = LLMAttributor(spy).attribute(_gap(), matrix, Ledger(root=tmp_path))
    assert spy.calls == 0
    assert hyp.verdict == "confirmed"          # heuristic handled it


def test_uncertain_calls_llm_and_returns_lead(tmp_path):
    spy = _SpyRuntime({"cause": "physics", "mechanism": "missing band-tail Urbach absorption",
                       "confidence": 0.6})
    hyp = LLMAttributor(spy).attribute(_gap(), _uncertain_matrix(), Ledger(root=tmp_path))
    assert spy.calls == 1
    assert hyp.cause == "physics"
    assert "Urbach" in hyp.mechanism
    assert hyp.verdict == "uncertain"          # ALWAYS a lead
    assert any("LLM novel-cause lead" in e for e in hyp.evidence_for)


def test_llm_failure_falls_back_to_heuristic(tmp_path):
    class _Boom:
        def complete(self, prompt, schema): raise RuntimeError("claude missing")
    hyp = LLMAttributor(_Boom()).attribute(_gap(), _uncertain_matrix(), Ledger(root=tmp_path))
    assert hyp.cause == "uncertain" and hyp.verdict == "uncertain"   # heuristic's uncertain


def test_refuted_mechanism_flagged(tmp_path):
    led = Ledger(root=tmp_path)
    led.add_negative(NegativeResult(approach="missing band-tail Urbach absorption",
                                    why_failed="x", evidence="y"))
    spy = _SpyRuntime({"cause": "physics", "mechanism": "missing band-tail Urbach absorption"})
    hyp = LLMAttributor(spy).attribute(_gap(), _uncertain_matrix(), led)
    assert hyp.verdict == "uncertain"
    assert any("refuted" in e.lower() for e in hyp.evidence_against)


def test_prompt_contains_context(tmp_path):
    led = Ledger(root=tmp_path)
    led.add_negative(NegativeResult(approach="DOS-cap projection", why_failed="x", evidence="y"))
    prompt = build_attribution_prompt(_gap(), _uncertain_matrix(), led)
    assert "Et_PVK ETL" in prompt                       # the gap
    assert "SOLARLAB_IFACE_PROJ" in prompt              # a matrix probe / flag menu
    assert "DOS-cap projection" in prompt               # the negatives ledger
    assert "JSON" in prompt
