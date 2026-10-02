"""Pair DC/AC consumers retain low words and reject copied or corrupted evidence."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import json

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_response as response
from perovskite_sim.experiments import one_dimensional_mechanism_r1_pair_response as pair
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.compensated import DD


def test_pair_inventory_resolves_a_residual_below_binary64_state_resolution(monkeypatch):
    fine = {"positive_m3": DD([1., 1.], [1e-18, 1e-18])}
    monkeypatch.setattr(pair, "require_state", lambda system, state: fine)
    system = SimpleNamespace(positive_nodes=np.array([0, 1]), dimension=2,
        positive_slice=slice(0, 2), widths=np.array([1., 1.]),
        positive_targets=np.array([2.]), ion_layout=SimpleNamespace(positive_components=([0, 1],)))
    row, derivative, residual = pair.inventory_rows(system, None)[0]
    assert row == 1
    np.testing.assert_array_equal(derivative, [.5, .5])
    assert residual == pytest.approx(1e-18, rel=1e-14)
    assert np.sum(fine["positive_m3"].hi)/2-1 == 0.


def test_pair_terminal_difference_keeps_nonzero_low_word_signal():
    def record(low):
        return {"precision_current_A_m2": {"hi": [1.], "lo": [low]}}
    plus, minus = record(2e-20), record(-2e-20)
    difference = (pair.terminal_current(plus)-pair.terminal_current(minus)).to_float()
    assert difference == 4e-20
    assert plus["precision_current_A_m2"]["hi"] == minus["precision_current_A_m2"]["hi"]
    assert float((pair.terminal_current(plus)-pair.terminal_current(plus)).to_float()) == 0.


def test_pair_physical_observation_retains_sub_high_word_displacement(monkeypatch):
    mat = SimpleNamespace(N_D=np.zeros(3), N_A=np.zeros(3), P_ion0=np.ones(3),
        poisson_factor=SimpleNamespace(C=np.ones(2)), ni_sq=np.ones(3),
        tau_n=np.ones(3), tau_p=np.ones(3), n1=np.ones(3), p1=np.ones(3),
        B_rad=np.zeros(3), C_n=np.zeros(3), C_p=np.zeros(3))
    system = SimpleNamespace(material=mat, widths=np.ones(3), interface_count=0, polarity=1.)
    def state(sign):
        fine = {"phi_V": DD([0., 1., 2.], [0., sign*1e-20, 0.]),
                "n_m3": DD(np.ones(3)), "p_m3": DD(np.ones(3)),
                "positive_m3": DD(np.ones(3)),
                "electron_current_A_m2": DD(np.zeros(2)), "hole_current_A_m2": DD(np.zeros(2)),
                "positive_flux_m2_s": DD(np.zeros(2))}
        return SimpleNamespace(fine=fine, n=np.ones(3), p=np.ones(3))
    monkeypatch.setattr(pair, "require_state", lambda system, state: state.fine)
    plus, minus = (pair.observations(system, state(sign))[0] for sign in (1, -1))
    np.testing.assert_array_equal(plus.hi, minus.hi)
    np.testing.assert_array_equal((plus-minus).to_float()[3], [-2e-20, -2e-20, 2e-20, 2e-20])


def test_pair_finite_difference_operands_retain_ionic_rate_and_storage_words():
    system = SimpleNamespace(positive_slice=slice(0, 1), positive_nodes=np.array([0]))
    def state(sign):
        return SimpleNamespace(rate=np.array([1.]), local_residual=np.array([0.]),
            fine={"positive_rate_m3_s": DD([1.], [sign*1e-20]),
                  "poisson_residual_C_m2": DD([0.], [0.])})
    up, down = state(1), state(-1)
    value = (pair.equation_vector(system, up)-pair.equation_vector(system, down)).to_float()
    assert value[0] == 2e-20
    assert (up.rate-down.rate)[0] == 0.


@pytest.mark.parametrize("mutation", ["schema", "representation", "response_numerics"])
def test_pair_numerical_contract_cannot_be_downgraded(mutation):
    record = pair.stamp({"schema": "R1ControlledDCResponseV1"})
    pair.validate_schema(record)
    record[mutation] = "legacy"
    with pytest.raises(ValueError, match="contract mismatch"):
        pair.validate_schema(record)


@pytest.fixture(scope="module")
def actual_pair_response():
    project = Path(__file__).resolve().parents[3]
    stack = load_device_from_yaml(project / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding = json.loads((project / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
    prepared = states.prepare_common_state(stack, 16, binding, backend="pair")
    zero = response.solve_controlled_dc(stack, 16, binding, prepared, backend="pair")
    biased = response.solve_controlled_dc(stack, 16, binding, prepared, backend="pair", voltage_V=.005)
    ac = response.small_signal_response(zero, [0., 1., 1e4])
    return stack, binding, prepared, zero, biased, ac


def test_actual_pair_dc_and_ac_retain_full_state_and_recompute(actual_pair_response):
    stack, binding, prepared, zero, biased, ac = actual_pair_response
    assert zero.evidence["schema"] == biased.evidence["schema"] == "R1ControlledDCResponseV2"
    assert zero.evidence["certified"] and biased.evidence["certified"]
    assert biased.evidence["terminal_current_A_m2"] > 0.
    assert zero.evidence["consumed_state_sha256"] == states.digest(zero.evidence["state"])
    assert any(np.any(value.lo != 0.) for value in zero.state.fine.values())
    rebuilt = response.restore_controlled_dc(states.json_data(zero.evidence), stack, 16, binding, prepared, backend="pair")
    response._same_response_content(rebuilt.evidence, zero.evidence)
    assert ac["schema"] == "R1ControlledSmallSignalV2"
    assert response.assess_small_signal_response(ac)["certified"]
    verification = response.verify_response_content(ac, stack=stack, intervals=16, binding=binding,
        prepared=prepared, request={"intervals": 16, "control": "D", "frequency_Hz": [0., 1., 1e4]}, backend="pair")
    assert verification["schema"] == "R1ResponseContentVerificationV2"
    assert verification["certified"] and verification["precision_state_and_observations_recomputed"]


@pytest.mark.parametrize("control", tuple("ABCD"))
def test_actual_controlled_dc_preserves_frozen_pair_populations(actual_pair_response, control):
    stack, binding, prepared, _, biased, _ = actual_pair_response
    dc = (biased if control == "D" else response.solve_controlled_dc(
        stack, 16, binding, prepared, backend="pair", control=control, voltage_V=.005))
    assert dc.evidence["control"] == control and dc.evidence["certified"]
    reference = prepared.to_dict()["state"]
    for frozen, field in ((control in "AC", "positive_m3"), (control in "AB", "occupancy")):
        if frozen:
            for word in ("hi", "lo"):
                np.testing.assert_array_equal(getattr(dc.state.fine[field], word), reference["precision_"+field+"_"+word])


@pytest.mark.parametrize("mutation", ["missing_low", "zero_low", "current_low", "copied_state", "legacy_schema", "policy"])
def test_actual_dc_restore_rejects_precision_loss_and_copy(actual_pair_response, mutation):
    stack, binding, prepared, zero, biased, _ = actual_pair_response
    record = states.json_data(copy.deepcopy(zero.evidence))
    if mutation == "missing_low":
        del record["state"]["precision_phi_V_lo"]
    elif mutation == "zero_low":
        key = next(name for name, value in record["state"].items() if name.endswith("_lo") and np.any(np.asarray(value) != 0.))
        record["state"][key] = np.zeros_like(record["state"][key]).tolist()
        record["consumed_state_sha256"] = states.digest(record["state"])
    elif mutation == "current_low":
        record["precision_current_A_m2"]["lo"][0] += 1e-30
    elif mutation == "copied_state":
        record["state"] = states.json_data(biased.evidence["state"])
        record["consumed_state_sha256"] = states.digest(record["state"])
    elif mutation == "legacy_schema":
        record["schema"] = "R1ControlledDCResponseV1"
    else:
        record["response_policy"]["maximum_newton_iterations"] += 1
    with pytest.raises(ValueError, match="mismatch|differs"):
        response.restore_controlled_dc(record, stack, 16, binding, prepared, backend="pair")


@pytest.mark.parametrize("requested", [
    {"intervals": 32, "control": "D", "voltage_V": 0.},
    {"intervals": 16, "control": "D", "voltage_V": .005},
    {"intervals": 16, "control": "D", "voltage_V": 0., "representation": "float64-baseline"},
])
def test_actual_replay_rejects_request_mismatch(actual_pair_response, requested):
    stack, binding, prepared, zero, _, _ = actual_pair_response
    with pytest.raises(ValueError, match="mismatch|differs"):
        response.verify_response_content(zero.evidence, stack=stack, intervals=16, binding=binding,
            prepared=prepared, request=requested, backend="pair")


def test_live_ac_rejects_tampered_low_words_before_reconstruction(actual_pair_response):
    _, _, _, zero, _, _ = actual_pair_response
    changed = copy.deepcopy(zero.state)
    original = changed.fine["phi_V"]
    lo = original.lo.copy()
    lo[1] += 1e-32
    changed.fine["phi_V"] = DD(original.hi, lo)
    with pytest.raises(ValueError, match="live_dc_state"):
        response.small_signal_response(replace(zero, state=changed), [0., 1., 1e4])


def test_actual_ac_replay_rejects_copied_derivative_values(actual_pair_response):
    stack, binding, prepared, _, _, ac = actual_pair_response
    copied = copy.deepcopy(ac)
    copied["derivative_levels"][-1]["state_per_V"] = copied["derivative_levels"][0]["state_per_V"].copy()
    # Keep all flags and the public curve intact: exact reconstruction still
    # binds the state coefficients to their declared derivative step.
    copied["derivative_levels"][-1]["state_per_V"][1, 0] += 1e-9
    with pytest.raises(ValueError, match="content mismatch"):
        response.verify_response_content(copied, stack=stack, intervals=16, binding=binding,
            prepared=prepared, request={"intervals": 16, "control": "D", "frequency_Hz": [0., 1., 1e4]}, backend="pair")


def test_actual_pair_tail_retains_low_words_and_unknown_infinite_tail(actual_pair_response):
    _, _, prepared, _, biased, _ = actual_pair_response
    tail = states.json_data(biased.evidence["state"])
    tail.update({key: biased.evidence[key] for key in ("prepared_sha256", "control", "voltage_V")})
    result = response.compare_transient_tail(biased, initial_state=prepared.to_dict()["state"], tail_state=tail,
        tail_regular_current_A_m2=biased.evidence["terminal_current_A_m2"], time_s=1.)
    assert result["schema"] == "R1TransientTailDCComparisonV2" and result["all_observables_agree"]
    assert result["infinite_tail_integral_bound_F_m2"] is None
    del tail["precision_positive_m3_lo"]
    with pytest.raises(ValueError, match="key coverage"):
        response.compare_transient_tail(biased, initial_state=prepared.to_dict()["state"], tail_state=tail,
            tail_regular_current_A_m2=biased.evidence["terminal_current_A_m2"], time_s=1.)


def pair_current_case(low=1., *, high=1e20):
    """Synthetic large offset with a resolvable low-word response of one unit."""
    from tests.unit.experiments.test_r1_v9_pair_codec import snapshot
    common = {"intervals": 3, "control": "D", "prepared_sha256": "prepared",
              "reference_sha256": "reference", "source": {"sha256": "source"}}
    times = [0., 1., 2.]
    step = {**common, "schema": "R1ControlledStepV2", "representation": "float64-pair-v1",
        "control_label": "D", "controls": {"nu_I": 1, "nu_t": 1},
        "times_s": times, "policy": {"refinement_substeps": [1, 2, 4]},
        "regular_currents": [{"report_contact_current_A_m2": [high, high]} for _ in times],
        "initial_event": {"impulse_charge_C_m2": 0.},
        "accepted_steps": [{"substeps": 4, "time_s": time, "regular_integrated_charge_C_m2": high*time} for time in times]}
    del step["control"]
    values = DD(np.full(6, high), np.full(6, low))
    observations = DD(np.vstack((values.hi, np.zeros((3, 6)))),
                      np.vstack((values.lo, np.zeros((3, 6)))))
    state = snapshot()
    dc = pair.stamp({**common, "schema": "R1ControlledDCResponseV1", "voltage_V": 0.,
        "certified": True, "state": state, "consumed_state_sha256": states.digest(state),
        "precision_current_A_m2": {"hi": values.hi.tolist(), "lo": values.lo.tolist()},
        "precision_observations": {"hi": observations.hi.tolist(), "lo": observations.lo.tolist()},
        "current_A_m2": values.to_float().tolist(), "terminal_current_A_m2": float(values[0].to_float()),
        "physical_face_labels": ["left_contact", "internal_0", "interface_left", "interface_right", "internal_2", "right_contact"]})
    return step, dc


def test_pair_current_and_integral_subtract_low_words_before_rounding():
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import (
        pair_step_current_charge_responses, compare_pair_step_current_charge)
    step, baseline = pair_current_case(1.)
    current, charge = pair_step_current_charge_responses(step, baseline, expected_times_s=[0., 1., 2.])
    np.testing.assert_array_equal(current.values, -np.ones((3, 2)))
    np.testing.assert_array_equal(charge.values, [0., -1., -2.])
    other_step, other_baseline = pair_current_case(-1.)
    assert baseline["current_A_m2"] == other_baseline["current_A_m2"]
    result = compare_pair_step_current_charge(step, other_step, baseline, other_baseline)
    assert not result["within_compared_budgets"]
    assert not result["regular_current"]["passed"] and not result["integrated_charge"]["passed"]
    assert result["regular_current"]["absolute_tolerance"] == 1e-7
    assert result["integrated_charge"]["absolute_tolerance"] == 1e-10


@pytest.mark.parametrize("mutation", ["legacy_step", "legacy_dc", "missing_low", "zeroed_low", "nonzero_bias", "control", "source", "prepared", "partial_axis", "duplicate_row"])
def test_pair_current_adapter_rejects_mismatched_or_incomplete_evidence(mutation):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import pair_step_current_charge_responses
    step, dc = pair_current_case()
    if mutation == "legacy_step":
        step["schema"] = "R1ControlledStepV1"
    elif mutation == "legacy_dc":
        dc["schema"] = "R1ControlledDCResponseV1"
    elif mutation == "missing_low":
        del dc["precision_current_A_m2"]["lo"]
    elif mutation == "zeroed_low":
        dc["precision_current_A_m2"]["lo"] = [0.]*6
    elif mutation == "nonzero_bias":
        dc["voltage_V"] = .005
    elif mutation in ("control", "source", "prepared"):
        key = "prepared_sha256" if mutation == "prepared" else mutation
        dc[key] = "different"
    elif mutation == "partial_axis":
        step["times_s"].pop(1)
        step["regular_currents"].pop(1)
    else:
        step["accepted_steps"].append(copy.deepcopy(step["accepted_steps"][1]))
    with pytest.raises(ValueError):
        pair_step_current_charge_responses(step, dc, expected_times_s=[0., 1., 2.])


def test_legacy_current_adapter_cannot_silently_drop_pair_baseline_words():
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import step_current_charge_responses
    step, dc = pair_current_case()
    with pytest.raises(ValueError, match="explicit pair DC baseline adapter"):
        step_current_charge_responses(step, [dc["current_A_m2"][0], dc["current_A_m2"][-1]])


def test_actual_dc_current_adapter_accepts_original_and_equal_preparation_artifacts(actual_pair_response):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import pair_step_current_charge_responses
    stack, binding, prepared, zero, _, _ = actual_pair_response
    del stack, binding
    step, _ = pair_current_case()
    step.update({key: zero.evidence[key] for key in ("intervals", "prepared_sha256", "reference_sha256", "source")})
    expected = pair_step_current_charge_responses(step, zero.evidence)
    # A different creation timestamp changes the artifact identity, while all
    # verified physical fields, pair words and executing-source identity agree.
    other = prepared.to_dict()
    other["created_utc"] = "2000-01-01T00:00:00+00:00"
    seed = other["seed_preparation"]
    seed["created_utc"] = other["created_utc"]
    seed["sha256"] = states.digest({key: value for key, value in seed.items() if key != "sha256"})
    other["seed_preparation_sha256"] = seed["sha256"]
    other["sha256"] = states.digest({key: value for key, value in other.items() if key != "sha256"})
    step["prepared_sha256"] = other["sha256"]
    actual = pair_step_current_charge_responses(step, zero.evidence, step_prepared=other, dc_prepared=prepared)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a.values, b.values)
    with pytest.raises(ValueError, match="preparation"):
        pair_step_current_charge_responses(step, zero.evidence, step_prepared=prepared, dc_prepared=other)


def test_actual_protocol_step_uses_pair_dc_current_adapter_without_field_aliases(actual_pair_response):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import run_r1_step
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import pair_step_current_charge_responses
    stack, binding, prepared, zero, _, _ = actual_pair_response
    step = run_r1_step(stack, 16, binding, prepared, control="D", times_s=[0., 1e-9], backend="pair")
    assert "control" not in step and step["control_label"] == "D"
    current, charge = pair_step_current_charge_responses(step, zero.evidence, expected_times_s=[0., 1e-9])
    assert current.values.shape == (2, 2) and charge.values.shape == (2,)
    changed = copy.deepcopy(step)
    changed["controls"]["nu_t"] = 0
    with pytest.raises(ValueError, match="control identity"):
        pair_step_current_charge_responses(changed, zero.evidence, expected_times_s=[0., 1e-9])
