"""Revision-six result verification uses equations on successful and failed data."""

from copy import deepcopy
import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as common
from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import read_json
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import run_r1_step, r1_policy
from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import verify_result_records
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.fixtures.r1_reference import approved_r1_binding
from tests.unit.experiments.test_one_dimensional_mechanism_r1_stage_one_cli import runner, PROJECT
from tests.unit.experiments.test_one_dimensional_mechanism_r1_evidence_revision_four import canonical_seal

pytestmark = pytest.mark.slow
FIXTURE = "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"


@pytest.fixture(scope="module")
def physical_case():
    binding = approved_r1_binding()
    stack = load_device_from_yaml(PROJECT / FIXTURE)
    prepared = common.prepare_common_state(stack, 16, binding, policy=r1_policy())
    result = run_r1_step(stack, 16, binding, prepared, times_s=[0., 1e-9],
                         policy=r1_policy(), physics_evidence=True)
    return prepared.to_dict(), result


@pytest.fixture
def bundle(physical_case, runner, tmp_path):
    prepared, result = physical_case
    for name, value in (("PreparedStateV1.json", prepared), ("ReferenceBindingV1.json", approved_r1_binding()),
                        ("StepResultV1.json", result), ("AcceptedStepsV1.json", result["accepted_steps"])):
        runner.write_json(tmp_path / name, value)
    (tmp_path / "SourceFixtureV1.yaml").write_bytes((PROJECT / FIXTURE).read_bytes())
    protocol = {key: common.json_data(result[key]) for key in ("intervals", "amplitude_V", "times_s", "policy")}
    protocol.update(control="D", nonlinear_factor=.1, time_substeps=[1, 2, 4])
    runner.write_json(tmp_path / "ProtocolV1.json", protocol)
    rows = result["accepted_steps"]
    finite = sum(row["dt_s"] > 0 for row in rows)
    completion = {"stage": "step", "status": "passed", "evidence_revision": 6,
        "accepted_record_count": len(rows), "observed_record_count": len(rows),
        "persisted_record_count": len(rows), "persisted_finite_step_count": finite,
        "physical_passed_finite_step_count": finite}
    return tmp_path, completion


def test_successful_current_record_has_no_range_only_physics_claims(bundle):
    report = verify_result_records(*bundle)
    assert report["scientifically_accepted"] and report["content_matches_recomputed"]
    assert report["physical_limits_satisfied"]
    assert report["range_only"] == []
    assert not report["integration_replayed"]


def fail_prefix(bundle, runner):
    output, completion = bundle
    result = read_json(output / "StepResultV1.json")
    (output / "StepResultV1.json").unlink()
    (output / "StepResultV1.npz").unlink(missing_ok=True)
    result.pop("sha256")
    result["accepted_steps"] = result["accepted_steps"][:2]
    for key in ("output_states", "regular_currents", "finite_step_averages", "accepted_state_arrays", "charge_integral"):
        result.pop(key)
    result["certificate"] = {"certified": False, "reasons": ["injected_stop"]}
    failure = {"type": "InjectedStop", "message": "bounded test interruption"}
    result["failure"] = failure
    completion.update(status="failed", failure=failure, accepted_record_count=2,
                      observed_record_count=2, persisted_record_count=2,
                      persisted_finite_step_count=1, physical_passed_finite_step_count=1)
    runner.write_json(output / "FailureV1.json", failure)
    runner._record_failure_result(output, result)
    runner.write_json(output / "AcceptedStepsV1.json", result["accepted_steps"])
    (output / "AcceptedStepsV1.npz").unlink(missing_ok=True)
    return result


def test_honest_failed_prefix_is_checked_without_changing_failed_status(bundle, runner):
    fail_prefix(bundle, runner)
    report = verify_result_records(*bundle)
    assert report["content_matches_recomputed"] and report["checked_row_count"] == 2
    assert not report["scientifically_accepted"] and not report["certified"]


def test_coordinated_failed_payload_and_saved_row_edits_are_rejected(bundle, runner):
    result = fail_prefix(bundle, runner)
    output, _ = bundle
    result["accepted_steps"][1]["state"]["n_m3"][1] *= 2.
    runner._record_failure_result(output, result)
    runner.write_json(output / "AcceptedStepsV1.json", result["accepted_steps"])
    with pytest.raises(ValueError, match="accepted state"):
        verify_result_records(*bundle)


def test_failed_sidecar_is_not_exempt_from_validation(bundle, runner):
    fail_prefix(bundle, runner)
    output, _ = bundle
    # Add a sidecar whose name maps to an actual numeric field, but whose
    # value no longer matches the failed JSON payload.
    np.savez_compressed(output / "FailedResultV1.npz", **{"data.times_s": np.array([0., 2e-9])})
    with pytest.raises(ValueError, match="failed result JSON/NPZ"):
        verify_result_records(*bundle)


def test_accepted_state_sidecar_is_checked_against_the_saved_state(bundle):
    output, _ = bundle
    rows = read_json(output / "AcceptedStepsV1.json")
    altered = np.asarray(rows[1]["state"]["n_m3"]) * 2.
    np.savez_compressed(output / "AcceptedStepsV1.npz", **{"data.1.state.n_m3": altered})
    with pytest.raises(ValueError, match="JSON/NPZ mismatch"):
        verify_result_records(*bundle)


def test_failed_case_cannot_hide_a_different_step_result(bundle, runner):
    result = fail_prefix(bundle, runner)
    output, _ = bundle
    result["amplitude_V"] *= 2.
    runner.write_json(output / "StepResultV1.json", result)
    with pytest.raises(ValueError, match="failed step duplicate"):
        verify_result_records(*bundle)


def test_failed_prepare_cannot_hide_an_existing_modified_prepared_state(bundle, runner):
    output, completion = bundle
    failure = {"type": "PostPreparationFailure", "message": "after prepared state write"}
    completion.update(stage="prepare", status="failed", failure=failure)
    runner.write_json(output / "FailureV1.json", failure)
    prepared = read_json(output / "PreparedStateV1.json")
    prepared["state"]["n_m3"][1] *= 2.
    runner.write_json(output / "PreparedStateV1.json", canonical_seal(prepared))
    with pytest.raises(ValueError, match="physical array"):
        verify_result_records(output, completion)
