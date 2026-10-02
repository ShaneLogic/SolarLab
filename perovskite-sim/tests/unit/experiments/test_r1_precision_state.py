"""Pair preparations preserve the old seed proof and add a new actual state."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_precision_state as pairs
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states


def seal(value):
    value = deepcopy(value)
    value.pop("sha256", None)
    value["sha256"] = states.digest(value)
    return value


@pytest.fixture
def harness(monkeypatch):
    calls, active = [], []
    stack, binding = {"device": "frozen"}, {"sha256": "binding", "f_ref": [0.4]}
    source = {"sha256": "source", "run_class": "development"}
    policy = {"limit": 1.0}
    seed_state = {"phi_V": [0.0, 0.1], "n_m3": [1., 2.], "p_m3": [2., 1.],
                  "positive_m3": [3., 3.], "occupancy": [0.4], "sheet_charge_C_m2": [0.],
                  "trace_potential_V": [[0.03, 0.07]], "trace_state_m3": [[1., 2., 3., 4.]]}
    seed_body = {
        "schema": "R1CommonStateV1", "kind": "equilibrium_D", "state_time": "0-",
        "source": source, "intervals": 16, "fixed_reference": binding,
        "reference_sha256": binding["sha256"], "physical_stack": stack,
        "stack_sha256": states.digest(stack), "preparation_controls": {"nu_I": 1, "nu_t": 1},
        "grid": {"tag": "original"}, "dc_state": {"tag": "healthy", "state": seed_state},
        "state": seed_state, "preparation_policy": policy,
        "qf_references_V": {"electron": [0., 0.], "hole": [0., 0.]},
        "preparation_checks": {"certified": True, "metrics": {"gate": 0.},
                               "limits": {"gate": 1.}, "reasons": []},
        "verification": {"schema": "R1PreparationVerificationV1"},
        "history": {"kind": "independent_D_dark_dc"}, "environment": {"python": "unit"},
        "created_utc": "2026-09-20T00:00:00+00:00", "study_spec_sha256": "study",
        "contact_velocities_m_s": {"left": 1.}, "contact_certificate": {"certified": True},
    }
    seed = states.R1PreparedState.from_dict(seal(seed_body))

    @contextmanager
    def legacy_context():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    def original_prepare(actual_stack, intervals, actual_binding, *, policy=None):
        assert active and actual_stack == stack and actual_binding == binding and intervals == 16
        calls.append("legacy_prepare")
        return seed

    def original_verify(prepared, actual_stack, actual_binding, *, policy=None):
        assert active
        record = prepared.to_dict()
        calls.append(("legacy_verify", policy))
        if (actual_stack != stack or actual_binding != binding
                or record["fixed_reference"] != binding or record["physical_stack"] != stack
                or record["preparation_controls"] != {"nu_I": 1, "nu_t": 1}
                or record["dc_state"]["tag"] != "healthy"):
            raise states.R1StateError("original seed physics or identity rejected")
        return SimpleNamespace(policy=policy, qfn_reference=np.zeros(2), qfp_reference=np.zeros(2),
                               controls=states.R1DynamicsControls(), grid=np.array([0., 1.])), seed_state

    def fine_factory(baseline, before):
        assert not active and before == seed_state
        calls.append("fine_factory")
        value = deepcopy(seed_state)
        # Actual precision arithmetic may change the main word as well.
        value["phi_V"][1] = 0.10000000000000002
        for name in pairs.REQUIRED_PAIRS:
            high = np.asarray(value.get(name, [0., 0.]))
            value["precision_" + name + "_hi"] = high.tolist()
            value["precision_" + name + "_lo"] = np.full(high.shape, 1e-30).tolist()
        return baseline, SimpleNamespace(snapshot=value, residual=0.)

    def checks(system, initial, actual_policy):
        assert not active
        calls.append("fine_checks")
        passed = initial.residual <= actual_policy["limit"]
        return {"metrics": {"gate": initial.residual}, "limits": {"gate": actual_policy["limit"]},
                "certified": passed, "reasons": [] if passed else ["gate"]}

    monkeypatch.setattr(states, "snapshot", lambda system, initial: deepcopy(initial.snapshot))
    monkeypatch.setattr(states, "equilibrium_checks", checks)
    monkeypatch.setattr(states, "execution_source", lambda: deepcopy(source))
    monkeypatch.setattr(states, "_preparation_policy", lambda value: value)
    prepare, verify = pairs.build_hooks(original_prepare, original_verify, legacy_context, fine_factory)
    return SimpleNamespace(prepare=prepare, verify=verify, stack=stack, binding=binding, policy=policy,
                           seed=seed, source=source, calls=calls, original_prepare=original_prepare,
                           original_verify=original_verify, legacy_context=legacy_context,
                           fine_factory=fine_factory)


def prepared(harness):
    return harness.prepare(harness.stack, 16, harness.binding, policy=harness.policy)


def test_actual_pair_state_can_differ_from_verified_seed_without_old_schema_claim(harness):
    pair = prepared(harness)
    record = pair.to_dict()
    assert isinstance(pair, states.R1PreparedState)
    assert record["schema"] == "R1CommonStatePairV1"
    assert record["seed_preparation"] == harness.seed.to_dict()
    assert record["seed_preparation_sha256"] == harness.seed.sha256
    assert record["dc_state"] == harness.seed.to_dict()["dc_state"]
    assert record["state"]["phi_V"] != record["seed_preparation"]["state"]["phi_V"]
    assert record["dc_state_role"] == pairs.DC_STATE_ROLE
    system, initial = harness.verify(pair, harness.stack, harness.binding)
    assert initial.snapshot == record["state"]
    assert harness.calls[:3] == ["legacy_prepare", ("legacy_verify", None), "fine_factory"]


def test_legacy_codec_rejects_pair_and_pair_codec_rejects_legacy(harness):
    pair = prepared(harness)
    with pytest.raises(states.R1StateError, match="schema"):
        states.R1PreparedState.from_dict(pair.to_dict())
    with pytest.raises(states.R1StateError, match="schema"):
        pairs.R1CommonStatePair.from_dict(harness.seed.to_dict())
    with pytest.raises(states.R1StateError, match="schema"):
        harness.verify(harness.seed, harness.stack, harness.binding)
    downgraded = pair.to_dict()
    downgraded["schema"] = "R1CommonStateV1"
    with pytest.raises(states.R1StateError, match="schema"):
        harness.verify(seal(downgraded), harness.stack, harness.binding)


def test_resealed_low_bit_tamper_fails_deterministic_reconstruction(harness):
    record = prepared(harness).to_dict()
    record["state"]["precision_phi_V_lo"][1] *= 2
    with pytest.raises(states.R1StateError, match="deterministic reconstruction"):
        harness.verify(seal(record), harness.stack, harness.binding)


def test_resealed_fine_checks_cannot_replace_actual_original_gate_result(harness):
    record = prepared(harness).to_dict()
    record["preparation_checks"]["metrics"]["gate"] = 0.25
    with pytest.raises(states.R1StateError, match="recomputed physics"):
        harness.verify(seal(record), harness.stack, harness.binding)


def test_missing_low_fields_or_changed_shapes_rejected_before_replay(harness):
    record = prepared(harness).to_dict()
    del record["state"]["precision_n_m3_lo"]
    with pytest.raises(states.R1StateError, match="matching high/low"):
        pairs.R1CommonStatePair.from_dict(seal(record))
    record = prepared(harness).to_dict()
    record["state"]["precision_phi_V_lo"] = [0.]
    with pytest.raises(states.R1StateError, match="invalid high/low"):
        pairs.R1CommonStatePair.from_dict(seal(record))


def test_changed_seed_source_or_controls_cannot_be_hidden_by_resealing(harness):
    for field, value in (("source", {"sha256": "other", "run_class": "development"}),
                         ("preparation_controls", {"nu_I": 0, "nu_t": 1})):
        record = prepared(harness).to_dict()
        record[field] = value
        with pytest.raises(states.R1StateError, match="seed identity field"):
            pairs.R1CommonStatePair.from_dict(seal(record))


def test_consistently_resealed_bad_seed_still_runs_original_verifier(harness):
    record = prepared(harness).to_dict()
    seed = record["seed_preparation"]
    seed["dc_state"]["tag"] = "bad"
    seed = seal(seed)
    record["seed_preparation"], record["seed_preparation_sha256"] = seed, seed["sha256"]
    record["dc_state"] = seed["dc_state"]
    with pytest.raises(states.R1StateError, match="original seed physics"):
        harness.verify(seal(record), harness.stack, harness.binding)


def test_source_reference_and_stack_are_checked_against_live_inputs(harness):
    pair = prepared(harness)
    with pytest.raises(states.R1StateError, match="original seed physics"):
        harness.verify(pair, harness.stack, {"sha256": "other"})
    with pytest.raises(states.R1StateError, match="original seed physics"):
        harness.verify(pair, {"device": "other"}, harness.binding)
    harness.source["sha256"] = "new_source"
    with pytest.raises(states.R1StateError, match="source"):
        harness.verify(pair, harness.stack, harness.binding)


def test_fine_gate_failure_retains_actual_seed_and_fine_snapshot(harness):
    def failing_factory(baseline, initial):
        system, actual = harness.fine_factory(baseline, initial)
        actual.residual = 2.
        return system, actual

    prepare, _ = pairs.build_hooks(harness.original_prepare, harness.original_verify,
                                   harness.legacy_context, failing_factory)
    with pytest.raises(states.R1StateError, match="original equilibrium gates") as caught:
        prepare(harness.stack, 16, harness.binding, policy=harness.policy)
    result = caught.value.result
    assert result["certified"] is False
    assert result["raw_preparation"]["seed_preparation"] == harness.seed.to_dict()
    assert result["raw_preparation"]["state"]["precision_phi_V_lo"]
    assert result["checks"]["metrics"]["gate"] == 2.


def test_stricter_consumer_failure_is_not_ignored(harness, monkeypatch):
    pair = prepared(harness)
    original = states.equilibrium_checks

    def consumer_checks(system, initial, policy):
        result = original(system, initial, policy)
        if policy["limit"] < 1.:
            result.update(certified=False, reasons=["stricter_gate"])
        return result

    monkeypatch.setattr(states, "equilibrium_checks", consumer_checks)
    with pytest.raises(states.R1StateError, match="consumer policy"):
        harness.verify(pair, harness.stack, harness.binding, policy={"limit": 0.1})


def test_exported_pair_and_seed_are_fresh_copies(harness):
    pair = prepared(harness)
    external = pair.to_dict()
    external["seed_preparation"]["source"]["sha256"] = "changed"
    external["state"]["precision_phi_V_lo"][0] = 7.
    actual = pair.to_dict()
    assert actual["seed_preparation"]["source"] == harness.source
    assert actual["state"]["precision_phi_V_lo"][0] == 1e-30


def test_fine_factory_cannot_silently_freeze_one_of_the_D_dynamics(harness):
    def wrong_control(baseline, before):
        system, state = harness.fine_factory(baseline, before)
        system.controls = states.R1DynamicsControls(nu_I=0, nu_t=1)
        return system, state

    prepare, _ = pairs.build_hooks(harness.original_prepare, harness.original_verify,
                                   harness.legacy_context, wrong_control)
    with pytest.raises(states.R1StateError, match="D controls"):
        prepare(harness.stack, 16, harness.binding, policy=harness.policy)
