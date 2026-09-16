"""Saved-state equation replay rejects coordinated changes to physical evidence."""

from copy import deepcopy

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from perovskite_sim.experiments.one_dimensional_mechanism_r1_physics_validation import (
    R1PhysicsValidationError, verify_r1_step_physics,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.fixtures.r1_reference import approved_r1_binding
from tests.integration.test_one_dimensional_mechanism_r1 import FIXTURE

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cases():
    stack, binding = load_device_from_yaml(FIXTURE), approved_r1_binding()
    policy = protocol.r1_policy()
    prepared = prepare_common_state(stack, 16, binding, policy=policy)
    records = {label: protocol.run_r1_step(stack, 16, binding, prepared, control=label,
        policy=policy, physics_evidence=True) for label in "ABCD"}
    return stack, binding, prepared, records


def verify(cases, record, **kwargs):
    stack, binding, prepared, _ = cases
    return verify_r1_step_physics(stack, 16, binding, prepared, record, **kwargs)


@pytest.mark.parametrize("label", list("ABCD"))
def test_actual_n16_controlled_states_reconstruct_all_equations(cases, label):
    report = verify(cases, cases[-1][label])
    assert report["certified"] and report["evidence_matches_equations"]
    assert report["checked_row_count"] == report["expected_row_count"] == 31
    assert report["metrics"] == cases[-1][label]["certificate"]["metrics"]
    assert report["provenance_only"] == ["Newton_iteration_path", "Newton_iteration_counts",
                                         "earlier_iterate_maximum_jacobian_nnz", "execution_environment"]
    for row in cases[-1][label]["accepted_steps"]:
        detail = row["physics_reconstruction"]
        assert detail["voltage_V"] == .005
        assert np.all(np.isfinite(detail["coordinate"]))
        for component in ("positive_ion_rate", "positive_ion_flux"):
            value = detail["eliminated_operator"][component]
            assert value["relative_error"] == value["maximum_absolute_difference"] / value["normalization_scale"]
            assert value["normalization_scale"] == max(value["direct_maximum_absolute"],
                value["eliminated_maximum_absolute"], value["normalization_floor"])
            np.testing.assert_array_equal(value["difference"], np.asarray(value["direct"]) - value["eliminated"])


def test_opt_in_recording_preserves_all_original_numerical_results(cases):
    stack, binding, prepared, records = cases
    fresh = prepare_common_state(stack, 16, binding, policy=protocol.r1_policy())
    legacy = protocol.run_r1_step(stack, 16, binding, fresh, policy=protocol.r1_policy())
    enhanced = records["D"]
    for collection in ("output_states", "accepted_state_arrays", "finite_step_averages"):
        for name, value in legacy[collection].items():
            if isinstance(value, str):
                assert enhanced[collection][name] == value
            else:
                np.testing.assert_array_equal(enhanced[collection][name], value)
    assert enhanced["certificate"] == legacy["certificate"]
    assert enhanced["charge_integral"] == legacy["charge_integral"]
    assert enhanced["regular_currents"] == legacy["regular_currents"]


@pytest.mark.parametrize("factor", [1e-3, -1.0, 1e3])
def test_coordinated_current_charge_and_diagnostic_scaling_rejected(cases, factor):
    forged = deepcopy(cases[-1]["D"])
    for row in forged["accepted_steps"]:
        if row["dt_s"] == 0:
            continue
        for section in (row["physical"], row["physics_reconstruction"]):
            for name in list(section):
                if "current" in name or name.endswith("_A_m2"):
                    if isinstance(section[name], (float, int, list)):
                        section[name] = (np.asarray(section[name]) * factor).tolist()
        row["regular_integrated_charge_C_m2"] *= factor
    for name in ("regular_by_substeps_C_m2", "complete_by_substeps_C_m2"):
        forged["charge_integral"][name] = {key: value * factor for key, value in forged["charge_integral"][name].items()}
    with pytest.raises(R1PhysicsValidationError, match="state equations|state-to-current"):
        verify(cases, forged)


def test_frozen_state_with_original_relaxation_current_rejected(cases):
    forged = deepcopy(cases[-1]["D"])
    reference = forged["accepted_steps"][0]["state"]
    for row in forged["accepted_steps"]:
        row["state"] = deepcopy(reference)
        row["physics_reconstruction"]["coordinate"] = [0.] * len(row["physics_reconstruction"]["coordinate"])
    for name in forged["output_states"]:
        forged["output_states"][name] = np.asarray([reference[name]] * len(forged["times_s"]))
    with pytest.raises(R1PhysicsValidationError, match="accepted state|state equations|state-to-current"):
        verify(cases, forged)


@pytest.mark.parametrize("metric", ["nonlinear_residual", "local_carrier_residual", "local_gauss_residual",
    "analytic_jacobian_error", "charge_balance_relative", "all_face_current_relative",
    "interface_current_relative", "eliminated_operator_error", "trap_storage_normalized_error"])
def test_certificate_metrics_cannot_be_forged_within_allowed_ranges(cases, metric):
    forged = deepcopy(cases[-1]["D"])
    original = forged["certificate"]["metrics"][metric]
    forged["certificate"]["metrics"][metric] = original * .5 if original else 1e-30
    with pytest.raises(R1PhysicsValidationError, match="recomputed physical certificate"):
        verify(cases, forged)


def test_trap_balance_and_nonlinear_claims_are_recomputed(cases):
    forged = deepcopy(cases[-1]["D"])
    row = forged["accepted_steps"][1]
    row["physical"]["trap_storage_error_A_m2"] = 0.
    row["physical"]["trap_storage_check"].update(error_A_m2=0., normalized_error=0.)
    row["scaled_nonlinear_residual"] = 0.
    row["physics_reconstruction"]["scaled_nonlinear_residual"] = 0.
    with pytest.raises(R1PhysicsValidationError, match="state equations|nonlinear residual|state-to-current"):
        verify(cases, forged)


def test_failed_prefix_is_auditable_but_never_scientifically_certified(cases):
    partial = deepcopy(cases[-1]["D"])
    partial["accepted_steps"] = partial["accepted_steps"][:3]
    partial["failure"] = {"type": "ExampleFailure", "message": "stopped after saved prefix"}
    partial["certificate"] = {"certified": False, "reasons": ["stopped after saved prefix"]}
    for key in ("output_states", "accepted_state_arrays", "regular_currents", "finite_step_averages", "charge_integral"):
        partial.pop(key)
    report = verify(cases, partial, allow_incomplete=True)
    assert report["evidence_matches_equations"]
    assert not report["complete"] and not report["certified"]
    assert report["checked_row_count"] == 3


@pytest.mark.parametrize("factor", [1e-3, -1., 1e3, 1.5])
def test_saved_physical_current_alone_is_bound_to_recomputed_state(cases, factor):
    forged = deepcopy(cases[-1]["D"])
    # Do not change state, diagnostics, charge integrals or the certificate:
    # this must exercise the isolated published-physical-block guard.
    current = forged["accepted_steps"][1]["physical"]["contact_maxwell_A_m2"]
    current[0] *= factor
    with pytest.raises(R1PhysicsValidationError, match="state-to-current and charge 1"):
        verify(cases, forged)


@pytest.mark.parametrize("field,value", [
    ("junction_polarity", 1.), ("execution_axes", {"intervals": 256, "time_substeps": [16, 32, 64]}),
    ("current_sign_convention", "forged sign"), ("scope", "full_r1_2_accepted"),
    ("scope_note", "all convergence passed"), ("version", "forged"),
])
def test_each_published_metadata_field_is_checked(cases, field, value):
    forged = deepcopy(cases[-1]["D"])
    forged[field] = value
    with pytest.raises(R1PhysicsValidationError, match="result metadata " + field):
        verify(cases, forged)


@pytest.mark.parametrize("field,key", [("finite_step_averages", "note"), ("charge_integral", "quadrature")])
def test_nested_result_meaning_cannot_be_relabelled(cases, field, key):
    forged = deepcopy(cases[-1]["D"])
    forged[field][key] = "trapezoidal point samples"
    with pytest.raises(R1PhysicsValidationError, match="meaning"):
        verify(cases, forged)


@pytest.mark.parametrize("nnz", [0, True, 1.5, 999999, 1])
def test_historical_nnz_must_be_sparse_integer_and_cover_saved_structure(cases, nnz):
    forged = deepcopy(cases[-1]["D"])
    forged["certificate"]["analytic_jacobian_nnz"] = nnz
    with pytest.raises(R1PhysicsValidationError, match="Jacobian"):
        verify(cases, forged)


@pytest.mark.parametrize("replacement", [None, {"passed": True}, {"passed": False, "nonfinite_numeric_paths": []}])
def test_finite_certificate_is_not_an_unchecked_provenance_field(cases, replacement):
    forged = deepcopy(cases[-1]["D"])
    if replacement is None:
        forged["certificate"].pop("finite_numeric_evidence")
    else:
        forged["certificate"]["finite_numeric_evidence"] = replacement
    with pytest.raises(R1PhysicsValidationError, match="finite numeric certificate"):
        verify(cases, forged)


def test_unclassified_result_field_and_nonfinite_value_cannot_hide_from_acceptance(cases):
    forged = deepcopy(cases[-1]["D"])
    forged["invented_physical_conclusion"] = True
    with pytest.raises(ValueError, match="unclassified controlled-step"):
        verify(cases, forged)
    forged.pop("invented_physical_conclusion")
    forged["accepted_steps"][1]["physical"]["contact_maxwell_A_m2"][0] = float("nan")
    with pytest.raises(R1PhysicsValidationError, match="nonfinite numeric evidence"):
        verify(cases, forged)


def partial_with_result_arrays(cases):
    partial = deepcopy(cases[-1]["D"])
    # Both coarser levels are complete; only 0+ of the finest level exists.
    partial["accepted_steps"] = partial["accepted_steps"][:15]
    partial["failure"] = {"type": "ExampleFailure", "message": "stopped after saved prefix"}
    partial["certificate"] = {"certified": False, "reasons": ["stopped after saved prefix"]}
    partial["output_states"] = {k: v[:1] for k, v in partial["output_states"].items()}
    partial["regular_currents"] = partial["regular_currents"][:1]
    partial["finite_step_averages"] = {k: v if k == "note" else v[:1]
                                        for k, v in partial["finite_step_averages"].items()}
    partial["accepted_state_arrays"] = {k: v[:15] for k, v in partial["accepted_state_arrays"].items()}
    integrals = {r["substeps"]: r["regular_integrated_charge_C_m2"] for r in partial["accepted_steps"]}
    partial["charge_integral"]["regular_by_substeps_C_m2"] = integrals
    impulse = partial["charge_integral"]["impulse_charge_C_m2"]
    partial["charge_integral"]["complete_by_substeps_C_m2"] = {k: impulse + v for k, v in integrals.items()}
    return partial


def test_existing_failed_prefix_arrays_are_checked_with_explicit_coverage(cases):
    report = verify(cases, partial_with_result_arrays(cases), allow_incomplete=True)
    assert report["content_matches_recomputed"] and not report["complete"]
    assert len(report["checked_result_fields"]) == 5
    assert report["unavailable_result_fields"] == []
    assert not report["certified"]


@pytest.mark.parametrize("field", ["output_states", "regular_currents", "finite_step_averages",
                                   "accepted_state_arrays", "charge_integral"])
def test_each_present_failed_prefix_result_block_rejects_forgery(cases, field):
    partial = partial_with_result_arrays(cases)
    if field == "output_states":
        partial[field]["n_m3"][0][0] *= 1.5
    elif field == "accepted_state_arrays":
        partial[field]["n_m3"][0][0] *= 1.5
    elif field == "finite_step_averages":
        partial[field]["internal_total_A_m2"][0][0] += 1e-3
    elif field == "charge_integral":
        partial[field]["impulse_charge_C_m2"] *= 7
    else:
        partial[field][0]["invented_current_A_m2"] = 1.
    with pytest.raises(R1PhysicsValidationError):
        verify(cases, partial, allow_incomplete=True)


def test_historical_rows_without_exact_coordinates_are_unavailable(cases):
    forged = deepcopy(cases[-1]["D"])
    forged["accepted_steps"][1].pop("physics_reconstruction")
    with pytest.raises(R1PhysicsValidationError, match="unavailable"):
        verify(cases, forged)


@pytest.mark.parametrize("field, value", [("time_s", 2e-9), ("dt_s", 2e-9)])
def test_step_duration_and_time_are_reconstructed(cases, field, value):
    forged = deepcopy(cases[-1]["D"])
    forged["accepted_steps"][1][field] = value
    with pytest.raises(R1PhysicsValidationError, match="accepted time|accepted duration"):
        verify(cases, forged)


def test_row_voltage_cannot_override_declared_excitation(cases):
    forged = deepcopy(cases[-1]["D"])
    forged["accepted_steps"][1]["physics_reconstruction"]["voltage_V"] = .01
    with pytest.raises(R1PhysicsValidationError, match="row voltage"):
        verify(cases, forged)


def test_reconstruction_failure_preserves_observed_state_and_coordinates(cases, monkeypatch):
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as physics
    stack, binding, _, _ = cases
    policy = protocol.r1_policy()
    prepared = prepare_common_state(stack, 16, binding, policy=policy)
    original = physics.capture_r1_physics_row

    def fail_after_acceptance(*args, **kwargs):
        if args[3] is not None:
            raise RuntimeError("diagnostic unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(physics, "capture_r1_physics_row", fail_after_acceptance)
    observed = []
    with pytest.raises(protocol.R1RunError) as caught:
        protocol.run_r1_step(stack, 16, binding, prepared, policy=policy,
                            times_s=[0., 1e-9], physics_evidence=True,
                            accepted_step_observer=observed.append)
    result = caught.value.result
    assert len(result["accepted_steps"]) == len(observed) == 2
    last = observed[-1]
    assert last["solver_accepted"] and not last["physical_checks_passed"]
    assert last["physics_reconstruction"]["available"] is False
    assert np.all(np.isfinite(last["physics_reconstruction"]["coordinate"]))
    assert last["state"]["n_m3"]
    assert result["certificate"]["reasons"] == ["physics_reconstruction_failed"]
