# tests/unit/autoloop/test_verify.py
from perovskite_sim.autoloop.types import Gap, AblationProbe, AblationMatrix, Hypothesis
from perovskite_sim.autoloop.verify import (
    SKEPTIC_LENSES, MultiSkepticVerifier, refute_prompt, gate_g5_verify,
)


def _gap():
    return Gap(id="trend:Et_PVK ETL:V_oc", metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _matrix():
    return AblationMatrix(gap_id="trend:Et_PVK ETL:V_oc", baseline_val=40.0, probes=(
        AblationProbe("SOLARLAB_IFACE_PROJ", "flag", {}, 40.0, 39.9, -0.1, True),))


def _lead():
    return Hypothesis(gap_id="trend:Et_PVK ETL:V_oc", cause="physics",
                      mechanism="missing band-tail Urbach absorption", verdict="uncertain")


class _ScriptedRuntime:
    """Returns a sequence of votes (or raises) keyed by call order."""
    def __init__(self, votes):
        self.votes = list(votes)
        self.calls = 0
    def complete(self, prompt, schema):
        v = self.votes[self.calls]
        self.calls += 1
        if isinstance(v, Exception):
            raise v
        return dict(v)


def test_all_fail_to_refute_confirms():
    rt = _ScriptedRuntime([{"refuted": False, "reason": "plausible"}] * 3)
    hyp = MultiSkepticVerifier(rt).verify(_lead(), _gap(), _matrix())
    assert hyp.verdict == "confirmed"
    assert hyp.verifier_votes == 3
    assert any("Urbach" in hyp.mechanism for _ in [0])


def test_majority_refute_refutes():
    rt = _ScriptedRuntime([{"refuted": True, "reason": "no support"},
                           {"refuted": True, "reason": "artifact"},
                           {"refuted": False, "reason": "maybe"}])
    hyp = MultiSkepticVerifier(rt).verify(_lead(), _gap(), _matrix())
    assert hyp.verdict == "refuted"
    assert any("artifact" in e for e in hyp.evidence_against)


def test_below_quorum_stays_uncertain():
    # only 1 of 3 skeptics succeeds (2 raise) -> < quorum(2) -> unchanged uncertain
    rt = _ScriptedRuntime([RuntimeError("x"), {"refuted": True, "reason": "r"}, RuntimeError("y")])
    hyp = MultiSkepticVerifier(rt).verify(_lead(), _gap(), _matrix())
    assert hyp.verdict == "uncertain"          # NOT refuted on a single errored-heavy run


def test_errored_skeptic_excluded_not_counted_as_refute():
    # 2 succeed (both fail-to-refute) + 1 errors -> quorum met, no refutes -> confirmed
    rt = _ScriptedRuntime([{"refuted": False, "reason": "ok"}, RuntimeError("x"),
                           {"refuted": False, "reason": "ok"}])
    hyp = MultiSkepticVerifier(rt).verify(_lead(), _gap(), _matrix())
    assert hyp.verdict == "confirmed" and hyp.verifier_votes == 2


def test_stringly_false_does_not_become_false_refute():
    # A live runtime returning {"refuted": "false"} must NOT be coerced to True
    # (bare bool("false") == True would wrongly refute a valid mechanism).
    rt = _ScriptedRuntime([{"refuted": "false", "reason": "plausible"}] * 3)
    hyp = MultiSkepticVerifier(rt).verify(_lead(), _gap(), _matrix())
    assert hyp.verdict == "confirmed"
    assert hyp.verifier_votes == 3


def test_stringly_true_refutes():
    rt = _ScriptedRuntime([{"refuted": "true", "reason": "no support"},
                           {"refuted": "true", "reason": "artifact"},
                           {"refuted": "false", "reason": "maybe"}])
    hyp = MultiSkepticVerifier(rt).verify(_lead(), _gap(), _matrix())
    assert hyp.verdict == "refuted"
    assert any("artifact" in e for e in hyp.evidence_against)


def test_unparseable_refuted_is_excluded_degrades_to_uncertain():
    # Two malformed verdicts are excluded (like errored skeptics); 1 valid
    # vote < quorum(2) -> hypothesis left uncertain, never a false refute.
    rt = _ScriptedRuntime([{"refuted": "maybe", "reason": "?"},
                           {"refuted": None, "reason": "?"},
                           {"refuted": True, "reason": "r"}])
    hyp = MultiSkepticVerifier(rt).verify(_lead(), _gap(), _matrix())
    assert hyp.verdict == "uncertain"


def test_refute_prompt_has_mechanism_and_lens():
    p = refute_prompt(_lead(), _gap(), _matrix(), "numerical-artifact")
    assert "Urbach" in p and "numerical-artifact" in p and "JSON" in p


def test_gate_g5_verify_maps_verdict():
    rt = _ScriptedRuntime([{"refuted": False, "reason": "ok"}] * 3)
    v = gate_g5_verify(_lead(), _gap(), _matrix(), MultiSkepticVerifier(rt))
    assert v.name == "G5_adversarial_verify" and v.passed is True


def test_lenses_default_is_three():
    assert len(SKEPTIC_LENSES) == 3
