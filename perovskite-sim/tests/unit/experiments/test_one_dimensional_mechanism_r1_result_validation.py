"""Real short trajectory cross-artifact validation and resealed mutations."""

import copy
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as common
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import check_zero_excitation, run_r1_step, r1_policy
from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import verify_result_records
from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import read_json
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.fixtures.r1_reference import approved_r1_binding
from tests.unit.experiments.test_one_dimensional_mechanism_r1_stage_one_cli import runner, PROJECT
from tests.unit.experiments.test_one_dimensional_mechanism_r1_evidence_revision_four import canonical_seal


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_step():
    stack = load_device_from_yaml(PROJECT / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding = approved_r1_binding()
    policy = r1_policy()
    prepared = common.prepare_common_state(stack, 16, binding, policy=policy)
    result = run_r1_step(stack, 16, binding, prepared, control="D", policy=policy)
    return prepared.to_dict(), result


@pytest.fixture
def saved_step(real_step, runner, tmp_path):
    prepared, original = real_step
    result = copy.deepcopy(original)
    rows = result["accepted_steps"]
    runner.write_json(tmp_path / "PreparedStateV1.json", prepared)
    runner.write_json(tmp_path / "StepResultV1.json", result)
    runner.write_json(tmp_path / "AcceptedStepsV1.json", rows)
    protocol = {key: common.json_data(result[key]) for key in ("intervals", "amplitude_V", "times_s", "policy")}
    protocol.update(control="D", nonlinear_factor=.1)
    runner.write_json(tmp_path / "ProtocolV1.json", protocol)
    completion = {
        "stage": "step", "status": "passed", "accepted_record_count": len(rows),
        "observed_record_count": len(rows), "persisted_record_count": len(rows),
        "persisted_finite_step_count": sum(row["phase"] == "accepted_regular_step" for row in rows),
        "physical_passed_finite_step_count": sum(row["phase"] == "accepted_regular_step" for row in rows),
    }
    return tmp_path, completion


def test_real_short_step_duplicate_representations_agree(saved_step):
    scope = verify_result_records(*saved_step)
    assert scope["integration_replayed"] is False
    assert scope["numerical_certificate_independently_approved"] is False
    assert "analytic_jacobian_error" in scope["range_only"]
    assert "storage_scale_m2_per_row" in scope["provenance_only"]


def test_new_producer_cannot_replace_contact_inclusive_metric_with_internal_only_value(saved_step, runner):
    path, completion = saved_step
    result = read_json(path / "StepResultV1.json")
    finest = max(result["policy"]["refinement_substeps"])
    spreads = []
    for row in result["accepted_steps"]:
        if row["substeps"] == finest:
            physical = row["physical"].get("regular_right_limit") or row["physical"]
            currents = np.asarray(physical["internal_maxwell_A_m2"])
            spreads.append(float(np.ptp(currents)) / max(float(np.max(np.abs(currents))), 1e-20))
    internal_only = max(spreads)
    assert internal_only < result["certificate"]["metrics"]["all_face_current_relative"]
    assert internal_only < result["certificate"]["limits"]["all_face_current_relative"]
    result["certificate"]["metrics"]["all_face_current_relative"] = internal_only
    reseal_and_sync_all_representations(path, result, runner)
    with pytest.raises(ValueError, match="certificate face current spread"):
        verify_result_records(path, completion)


def test_initial_right_limit_and_old_sources_keep_internal_only_metric():
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import _saved_face_current_spread
    physical = {"internal_maxwell_A_m2": [1., 1.], "contact_maxwell_A_m2": [1., 2.]}
    initial = {"dt_s": 0., "physical": {"regular_right_limit": physical}}
    finite = {"dt_s": 1., "physical": physical}
    assert _saved_face_current_spread(initial, include_contacts=True) == 0.
    assert _saved_face_current_spread(finite, include_contacts=False) == 0.
    assert _saved_face_current_spread(finite, include_contacts=True) == .5


def test_legacy_metric_scope_comes_from_bound_producer_bytes(tmp_path):
    import hashlib
    import zipfile
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_dynamics as dynamics
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import _legacy_contact_closure_scope
    relative = "experiments/one_dimensional_mechanism_r1_dynamics.py"
    current = Path(dynamics.__file__).read_bytes()
    record = {"source": {"files": {relative: hashlib.sha256(current).hexdigest()}}}
    assert _legacy_contact_closure_scope(record, tmp_path) is True
    old = b"class ControlledPhysicalInterfaceIonSystem:\n    pass\n"
    record["source"]["files"][relative] = hashlib.sha256(old).hexdigest()
    with pytest.raises(ValueError, match="requires its recorded producer source"):
        _legacy_contact_closure_scope(record, tmp_path)
    with zipfile.ZipFile(tmp_path / "SourceV1.zip", "w") as archive:
        archive.writestr("perovskite-sim/perovskite_sim/" + relative, old)
    assert _legacy_contact_closure_scope(record, tmp_path) is False
    record["source"]["files"][relative] = "0" * 64
    with pytest.raises(ValueError, match="producer differs from recorded source identity"):
        _legacy_contact_closure_scope(record, tmp_path)


@pytest.mark.parametrize("mutation", ["metric", "row", "array", "source", "impulse", "policy", "time", "control", "count", "sidecar"])
def test_resealed_result_changes_are_rejected(saved_step, runner, mutation):
    path, completion = saved_step
    result = read_json(path / "StepResultV1.json")
    if mutation == "metric":
        result["certificate"]["metrics"]["full_gauss_normalized"] = 0.0
    elif mutation == "row":
        result["accepted_steps"][-1]["physical_checks_passed"] = False
    elif mutation == "array":
        result["output_states"]["n_m3"][-1][1] *= 2
    elif mutation == "source":
        result["source"]["source_commit"] = "0" * 40
    elif mutation == "impulse":
        result["charge_integral"]["impulse_charge_C_m2"] *= 2
        # Retain internal digest: even an otherwise unconstrained field is bound.
        runner.write_json(path / "StepResultV1.json", result)
        with pytest.raises(ValueError, match="payload digest"):
            verify_result_records(path, completion)
        return
    elif mutation == "policy":
        result["policy"]["maximum_scaled_nonlinear_residual"] *= 2
    elif mutation == "time":
        result["times_s"][-1] *= 2
    elif mutation == "control":
        result["control_label"] = "A"
    elif mutation == "count":
        completion["persisted_record_count"] -= 1
    elif mutation == "sidecar":
        with np.load(path / "StepResultV1.npz", allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["data.output_states.n_m3"] *= 2
        np.savez_compressed(path / "StepResultV1.npz", **arrays)
    # The attacker can recalculate the result's own digest, too.
    if mutation != "sidecar":
        runner.write_json(path / "StepResultV1.json", canonical_seal(result))
    with pytest.raises(ValueError, match="result|certificate"):
        verify_result_records(path, completion)


def reseal_and_sync_all_representations(path, result, runner):
    """Grant the attacker all self-hashes and every matching JSON/NPZ copy."""
    with np.load(path / "StepResultV1.npz", allow_pickle=False) as data:
        names = list(data.files)
    arrays = {}
    for name in names:
        value = result
        for part in name.split(".")[1:]:
            value = value[int(part)] if isinstance(value, list) else value[part]
        arrays[name] = np.asarray(value)
    runner.write_json(path / "StepResultV1.json", canonical_seal(result))
    runner.write_json(path / "AcceptedStepsV1.json", result["accepted_steps"])
    np.savez_compressed(path / "StepResultV1.npz", **arrays)


@pytest.mark.parametrize("metric", [
    "nonlinear_residual", "local_carrier_residual", "local_gauss_residual",
    "all_face_current_relative", "inventory_relative_drift", "refinement_state_change",
    "refinement_current_change", "full_gauss_normalized", "full_charge_balance_normalized",
    "physical_current_spread_relative", "trap_storage_normalized_error",
])
def test_resealed_and_npz_synced_metric_forgery_is_rejected(saved_step, runner, metric):
    path, completion = saved_step
    result = read_json(path / "StepResultV1.json")
    value = result["certificate"]["metrics"][metric]
    result["certificate"]["metrics"][metric] = 0.0 if value else 1e-30
    reseal_and_sync_all_representations(path, result, runner)
    with pytest.raises(ValueError, match="result|certificate"):
        verify_result_records(path, completion)


@pytest.mark.parametrize("mutation", [
    "all_metrics_zero", "voltage", "controls", "complete_charge", "regular_charge",
    "impulse", "site", "frozen_checks", "reported_polarity", "current_decomposition",
    "trap_summary", "trap_budget", "row_integral",
])
def test_fully_resealed_cross_field_forgery_is_rejected(saved_step, runner, mutation):
    path, completion = saved_step
    result = read_json(path / "StepResultV1.json")
    if mutation == "all_metrics_zero":
        for name in result["certificate"]["metrics"]:
            if name not in {"full_gauss_normalized", "full_charge_balance_normalized", "physical_current_spread_relative"}:
                result["certificate"]["metrics"][name] = 0.0
    elif mutation == "voltage":
        result["voltage_V"] = [1234.0] * len(result["times_s"])
    elif mutation == "controls":
        result["controls"] = {"nu_I": 0, "nu_t": 0}
    elif mutation in {"complete_charge", "regular_charge"}:
        key = "complete_by_substeps_C_m2" if mutation == "complete_charge" else "regular_by_substeps_C_m2"
        result["charge_integral"][key] = {key: 1234.0 for key in result["charge_integral"][key]}
    elif mutation == "impulse":
        result["charge_integral"]["impulse_charge_C_m2"] *= 2
        result["initial_event"]["impulse_charge_C_m2"] *= 2
    elif mutation == "site":
        result["certificate"]["site_occupancy_fraction"] = 2.0
    elif mutation == "frozen_checks":
        result["certificate"]["exact_freeze_checks"] = {"positive_population": True}
    elif mutation == "reported_polarity":
        result["regular_currents"][-1]["report_contact_current_A_m2"][0] *= -1
    elif mutation == "current_decomposition":
        result["accepted_steps"][-1]["physical"]["contact_conduction_A_m2"][0] += 1234.0
    elif mutation == "trap_summary":
        result["certificate"]["trap_storage"]["maximum_normalized_error"] = 0.0
    elif mutation == "trap_budget":
        result["accepted_steps"][-1]["physical"]["trap_storage_check"]["charge_error_limit_C_m2"] = 1e30
    else:
        result["accepted_steps"][-1]["regular_integrated_charge_C_m2"] = 1234.0
    reseal_and_sync_all_representations(path, result, runner)
    with pytest.raises(ValueError, match="result|certificate"):
        verify_result_records(path, completion)


def test_unsaved_solver_diagnostic_is_not_claimed_independently_verified(saved_step, runner):
    path, completion = saved_step
    result = read_json(path / "StepResultV1.json")
    result["certificate"]["metrics"]["analytic_jacobian_error"] = 0.0
    reseal_and_sync_all_representations(path, result, runner)
    scope = verify_result_records(path, completion)
    assert "analytic_jacobian_error" in scope["range_only"]
    assert scope["numerical_certificate_independently_approved"] is False


@pytest.mark.parametrize("control", ["A", "B", "C"])
def test_real_frozen_controls_preserve_their_declared_populations(real_step, saved_step, runner, control):
    prepared, _ = real_step
    path, completion = saved_step
    stack = load_device_from_yaml(PROJECT / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    result = run_r1_step(stack, 16, approved_r1_binding(), prepared,
                         control=control, times_s=(0.0, 1e-9), policy=r1_policy())
    runner.write_json(path / "StepResultV1.json", result)
    runner.write_json(path / "AcceptedStepsV1.json", result["accepted_steps"])
    protocol = {key: common.json_data(result[key]) for key in ("intervals", "amplitude_V", "times_s", "policy")}
    protocol.update(control=control, nonlinear_factor=.1)
    runner.write_json(path / "ProtocolV1.json", protocol)
    count = len(result["accepted_steps"])
    finite = sum(row["phase"] == "accepted_regular_step" for row in result["accepted_steps"])
    completion.update(accepted_record_count=count, observed_record_count=count, persisted_record_count=count,
                      persisted_finite_step_count=finite, physical_passed_finite_step_count=finite)
    verify_result_records(path, completion)


def test_failed_missing_rows_cannot_claim_persisted_evidence(tmp_path, runner):
    failure = {"certified": False, "reason": "persistence failed before first write"}
    runner.write_json(tmp_path / "FailureV1.json", failure)
    completion = {"status": "failed", "stage": "step", "failure": failure,
                  "accepted_record_count": 0, "observed_record_count": 1,
                  "persisted_record_count": 0, "persisted_finite_step_count": 0,
                  "physical_passed_finite_step_count": 0}
    verify_result_records(tmp_path, completion)
    for key in ("accepted_record_count", "persisted_record_count", "persisted_finite_step_count",
                "physical_passed_finite_step_count"):
        with pytest.raises(ValueError, match="completion count"):
            verify_result_records(tmp_path, {**completion, key: 10000})


@pytest.fixture(scope="module")
def real_zero(real_step):
    prepared, _ = real_step
    stack = load_device_from_yaml(PROJECT / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    # Source identity is bound independently by the bundle verifier.
    record = copy.deepcopy(prepared)
    record["source"] = common.execution_source()
    record = canonical_seal(record)
    return record, check_zero_excitation(stack, 16, approved_r1_binding(), record)


@pytest.mark.parametrize("mutation", [None, "metrics", "impulse", "controls"])
def test_zero_control_specific_equations_and_no_impulse_are_recomputed(real_zero, tmp_path, runner, mutation):
    prepared, original = real_zero
    result = copy.deepcopy(original)
    if mutation == "metrics":
        result["controls"]["A"]["remaining_equations"]["metrics"] = {
            key: 1e30 for key in result["controls"]["A"]["remaining_equations"]["metrics"]}
    elif mutation == "impulse":
        result["controls"]["D"]["initial_event"]["impulse_charge_C_m2"] = 1234.0
    elif mutation == "controls":
        result["controls"]["D"]["controls"] = {"nu_I": 0, "nu_t": 0}
    runner.write_json(tmp_path / "PreparedStateV1.json", prepared)
    runner.write_json(tmp_path / "ZeroExcitationV1.json", canonical_seal(result))
    runner.write_json(tmp_path / "ProtocolV1.json", {"control": "ABCD", "nonlinear_factor": .1,
                      "policy": common.json_data(r1_policy())})
    completion = {"stage": "zero-check", "status": "passed"}
    if mutation is None:
        verify_result_records(tmp_path, completion)
    else:
        with pytest.raises(ValueError, match="zero-check"):
            verify_result_records(tmp_path, completion)
